#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
emit_tensix_cb_formats.py -- compile-time circular-buffer (CB) descriptor generator.

Foundry compiles each compute kernel once per TRISC thread (unpack, math, pack). Every
compile includes a generated header (ft_tensix_cb_formats.h) that bakes in, per CB index,
the data formats and tile geometry the three threads need. This tool writes that header
from a .cbtable file (or a uniform Float16_b default). Output is plain C, no includes.

Format codes are the hardware data-format codes from the public ISA documentation
(tt-isa-documentation, data-format and tile pages); everything else is Foundry-defined.

Status: Complete for the formats modelled (Float32, Float16, Tf32, Float16_b, Bfp8_b, Bfp4_b,
Bfp2_b).
Issues: None
Todo: None
"""

import argparse
import re
import sys
from pathlib import Path

TOOL_NAME = "emit_tensix_cb_formats"
MAILBOX_HEADER_RELATIVE_PATH = "../ft_tt_mailbox.h"
CB_COUNT_PATTERN = re.compile(r"^\s*#\s*define\s+FT_TT_NUM_CIRCULAR_BUFFERS\s+(\d+)u?\b", re.M)

FORMAT_CODES = {"Float32": 0, "Float16": 1, "Tf32": 4, "Float16_b": 5, "Bfp8_b": 6, "Bfp4_b": 7,
                "Bfp2_b": 15}
INVALID_FORMAT_CODE = 0xFF
BYTES_PER_ELEMENT = {"Float32": 4, "Tf32": 4, "Float16": 2, "Float16_b": 2}
MANTISSA_BITS = {"Bfp8_b": 8, "Bfp4_b": 4, "Bfp2_b": 2}
B_EXPONENT_FAMILY = {"Float16_b", "Bfp8_b", "Bfp4_b", "Bfp2_b"}
THIRTY_TWO_BIT_FORMATS = {"Float32", "Int32"}

SETTING_DEFAULTS = {"math_fidelity": 4, "approx": 0, "dst_accum_mode": 0}
GEOMETRY_DEFAULTS = (32, 32, 16, 4, 0, 0)
GEOMETRY_FIELD_NAMES = ("tile_rows", "tile_cols", "face_rows", "num_faces",
                        "partial_face", "narrow_tile")
SHARED_EXPONENT_ROW_ROUNDING = 16
FACE_COLUMNS = 16
UINT8_MAX = 255
UINT16_MAX = 65535

# Array suffix -> C type and default for an index that describes no CB.
GEOMETRY_ARRAY_SUFFIXES = ("num_faces", "partial_face", "face_rows", "narrow_tile",
                           "tile_rows", "tile_cols")

# Column key -> the C array suffix kernels use. The names keep one fixed sort order across
# the arrays: GCC's whole-program optimiser can order spill slots by symbol name, so renaming
# arrays out of that order moved two stack slots in one image (measured 2026-10-08, gdn).
PUBLIC_ARRAY_SUFFIX = {"num_faces": "tile_num_faces", "face_rows": "tile_face_rows",
                       "tile_bytes": "tile_size", "face_grid_rows": "num_face_rows",
                       "face_grid_cols": "num_face_cols"}


class CbTableError(Exception):
    """A malformed or unsupported input; reported as `<tool>: <message>`, exit 2."""


def read_circular_buffer_count(tool_directory):
    """Return N from `#define FT_TT_NUM_CIRCULAR_BUFFERS <digits>[u]` in ft_tt_mailbox.h."""
    header_path = tool_directory / MAILBOX_HEADER_RELATIVE_PATH
    try:
        header_text = header_path.read_text()
    except OSError as error:
        raise CbTableError(f"cannot read {header_path}: {error}")
    match = CB_COUNT_PATTERN.search(header_text)
    if not match:
        raise CbTableError(f"FT_TT_NUM_CIRCULAR_BUFFERS not defined in {header_path}")
    return int(match.group(1))


def parse_integer(token, context):
    try:
        return int(token, 0)
    except ValueError:
        raise CbTableError(f"{context}: '{token}' is not an integer")


def make_cb_description(index, format_name, geometry):
    """Validate one CB line and derive its face grid and tile byte count."""
    if format_name not in FORMAT_CODES:
        raise CbTableError(f"cb {index}: format '{format_name}' is not modelled")
    fields = dict(zip(GEOMETRY_FIELD_NAMES, geometry))
    if fields["tile_cols"] % FACE_COLUMNS != 0 or fields["tile_cols"] <= 0:
        raise CbTableError(f"cb {index}: tile_cols {fields['tile_cols']} is not a multiple of 16")
    face_cols = min(fields["tile_cols"] // FACE_COLUMNS, fields["num_faces"])
    if face_cols == 0 or fields["num_faces"] % face_cols != 0:
        raise CbTableError(f"cb {index}: num_faces {fields['num_faces']} does not tile "
                           f"{face_cols} face column(s)")
    fields.update(index=index, format=format_name, face_grid_cols=face_cols,
                  face_grid_rows=fields["num_faces"] // face_cols,
                  tile_bytes=compute_tile_bytes(format_name, fields))
    for name in GEOMETRY_FIELD_NAMES + ("face_grid_cols", "face_grid_rows"):
        if not 0 <= fields[name] <= UINT8_MAX:
            raise CbTableError(f"cb {index}: {name} {fields[name]} does not fit uint8_t")
    if not 0 <= fields["tile_bytes"] <= UINT16_MAX:
        raise CbTableError(f"cb {index}: tile bytes {fields['tile_bytes']} do not fit uint16_t")
    return fields


def compute_tile_bytes(format_name, fields):
    """Block floats: packed mantissas plus one shared exponent per face row (rounded to 16)."""
    elements = fields["tile_rows"] * fields["tile_cols"]
    if format_name in MANTISSA_BITS:
        exponent_rows = fields["face_rows"] * fields["num_faces"]
        rounded_rows = -(-exponent_rows // SHARED_EXPONENT_ROW_ROUNDING) * SHARED_EXPONENT_ROW_ROUNDING
        return elements * MANTISSA_BITS[format_name] // 8 + rounded_rows
    return elements * BYTES_PER_ELEMENT[format_name]


def parse_cb_line(arguments, descriptions, circular_buffer_count):
    if len(arguments) < 2:
        raise CbTableError("cb needs at least <index> <format>")
    index = parse_integer(arguments[0], "cb index")
    if not 0 <= index < circular_buffer_count:
        raise CbTableError(f"cb index {index} out of range (0..{circular_buffer_count - 1})")
    if index in descriptions:
        raise CbTableError(f"cb index {index} given twice")
    if len(arguments) - 2 > len(GEOMETRY_DEFAULTS):
        raise CbTableError(f"cb {index}: too many fields")
    given = [parse_integer(token, f"cb {index}") for token in arguments[2:]]
    geometry = tuple(given) + GEOMETRY_DEFAULTS[len(given):]
    descriptions[index] = make_cb_description(index, arguments[1], geometry)


def parse_cb_table(text, circular_buffer_count):
    """Return (settings, {index: description}) from cbtable text."""
    settings = dict(SETTING_DEFAULTS)
    descriptions = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        tokens = raw_line.split("#", 1)[0].split()
        if not tokens:
            continue
        directive, arguments = tokens[0], tokens[1:]
        try:
            if directive in settings:
                if len(arguments) != 1:
                    raise CbTableError(f"{directive} needs exactly one value")
                settings[directive] = parse_integer(arguments[0], directive)
            elif directive == "cb":
                parse_cb_line(arguments, descriptions, circular_buffer_count)
            else:
                raise CbTableError(f"unknown directive '{directive}'")
        except CbTableError as error:
            raise CbTableError(f"line {line_number}: {error}")
    if not descriptions:
        raise CbTableError("table has no cb line")
    return settings, descriptions


def make_uniform_default(circular_buffer_count):
    descriptions = {index: make_cb_description(index, "Float16_b", GEOMETRY_DEFAULTS)
                    for index in range(circular_buffer_count)}
    return dict(SETTING_DEFAULTS), descriptions


def choose_intermediate_format(settings, descriptions):
    """The 16-bit format that Float32 buffers pass through between unpack and pack."""
    ordered = [descriptions[index] for index in sorted(descriptions)]
    if settings["dst_accum_mode"] and all(d["format"] == "Float32" for d in ordered):
        return "Tf32"
    for description in ordered:
        if description["format"] not in ("Float32", "Tf32"):
            return "Float16_b" if description["format"] in B_EXPONENT_FAMILY else "Float16"
    return "Float16_b"


def derive_role_formats(format_name, intermediate_format, dst_accum_mode):
    """(unpack source, unpack destination, pack source, pack destination)."""
    if format_name in MANTISSA_BITS:
        return (format_name,) * 4
    if format_name == "Float32":
        pack_source = "Float32" if dst_accum_mode else intermediate_format
        return ("Float32", intermediate_format, pack_source, "Float32")
    return (format_name,) * 4


def format_array(c_type, name, values, circular_buffer_count):
    joined = ",".join(str(value) for value in values)
    assert len(values) == circular_buffer_count
    return f"FT_CB_TABLE_CONST {c_type} {name}[{circular_buffer_count}] = {{\n    {joined}\n}};\n"


def build_columns(descriptions, roles, circular_buffer_count):
    """Per-index column lists, keyed by field name, with unused-index defaults filled in."""
    def column(getter, default):
        return [getter(i) if i in descriptions else default for i in range(circular_buffer_count)]

    columns = {}
    for name in GEOMETRY_ARRAY_SUFFIXES + ("tile_bytes", "face_grid_rows", "face_grid_cols"):
        columns[name] = column(lambda i, n=name: descriptions[i][n], 0)
    for position, label in enumerate(("unpack_src", "unpack_dst", "pack_src", "pack_dst")):
        columns[label] = column(lambda i, p=position: FORMAT_CODES[roles[i][p]], INVALID_FORMAT_CODE)
    return columns


def emit_stage_arrays(prefix, destination_key, columns, circular_buffer_count):
    """Arrays for one stage ('unpack' or 'pack'); the source-format array is emitted by the caller."""
    text = format_array("uint8_t", f"ft_cb_{prefix}_dst_format", columns[destination_key],
                        circular_buffer_count)
    for suffix in GEOMETRY_ARRAY_SUFFIXES:
        text += format_array("uint8_t", f"ft_cb_{prefix}_{PUBLIC_ARRAY_SUFFIX.get(suffix, suffix)}", columns[suffix],
                             circular_buffer_count)
    text += format_array("uint16_t", f"ft_cb_{prefix}_tile_size", columns["tile_bytes"],
                         circular_buffer_count)
    for suffix in ("face_grid_rows", "face_grid_cols"):
        text += format_array("uint8_t", f"ft_cb_{prefix}_{PUBLIC_ARRAY_SUFFIX.get(suffix, suffix)}", columns[suffix],
                             circular_buffer_count)
    return text


def has_any_32bit_unpack(descriptions, roles):
    return any(roles[i][0] in THIRTY_TWO_BIT_FORMATS and roles[i][1] in THIRTY_TWO_BIT_FORMATS
               for i in descriptions)


def render_header(settings, descriptions, circular_buffer_count):
    intermediate = choose_intermediate_format(settings, descriptions)
    roles = {i: derive_role_formats(d["format"], intermediate, settings["dst_accum_mode"])
             for i, d in descriptions.items()}
    columns = build_columns(descriptions, roles, circular_buffer_count)
    count = circular_buffer_count
    source_array = format_array("uint8_t", "ft_cb_unpack_src_format", columns["unpack_src"], count)
    lines = ["#pragma once", "", "#define FT_CB_TABLE_CONST static const", ""]
    lines.append("#if defined(FT_TRISC_MATH) || defined(FT_TRISC_PACK)")
    lines.append(f"#define FT_CB_MATH_FIDELITY {settings['math_fidelity']}")
    lines.append(f"#define FT_CB_APPROX {1 if settings['approx'] else 0}")
    lines.append("#endif\n")
    lines.append("#if !defined(FT_TRISC_PACK)")
    lines.append(source_array + emit_stage_arrays("unpack", "unpack_dst", columns, count)
                 + "#endif\n")
    lines.append("#if !defined(FT_TRISC_MATH) && !defined(FT_TRISC_UNPACK)")
    lines.append(source_array)
    lines.append(format_array("uint8_t", "ft_cb_pack_src_format", columns["pack_src"], count)
                 + emit_stage_arrays("pack", "pack_dst", columns, count) + "#endif\n")
    lines.append("#if defined(FT_TRISC_MATH) || defined(FT_TRISC_PACK) || defined(FT_TRISC_UNPACK)")
    lines.append(f"#define FT_CB_DST_ACCUM_FP32 {1 if settings['dst_accum_mode'] else 0}")
    lines.append("#endif\n")
    lines.append(f"#define FT_CB_ANY_32BIT_UNPACK {1 if has_any_32bit_unpack(descriptions, roles) else 0}")
    return "\n".join(lines) + "\n"


def summarize(descriptions, uniform, circular_buffer_count):
    if uniform:
        return f"uniform Float16_b 32x32 4-face 2048 B on all {circular_buffer_count} CBs"
    return " ".join(f"{i}:{descriptions[i]['format']}/{descriptions[i]['tile_bytes']}B"
                    for i in sorted(descriptions))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n")[1])
    parser.add_argument("-o", "--output", required=True, help="header file to write")
    parser.add_argument("--cb-table", help="input .cbtable file")
    parser.add_argument("--uniform-default", action="store_true",
                        help="describe every CB as Float16_b 32x32")
    arguments = parser.parse_args(argv)
    if bool(arguments.cb_table) == arguments.uniform_default:
        raise CbTableError("give exactly one of --cb-table or --uniform-default")
    return arguments


def main(argv):
    try:
        arguments = parse_arguments(argv)
        count = read_circular_buffer_count(Path(__file__).resolve().parent)
        if arguments.uniform_default:
            settings, descriptions = make_uniform_default(count)
        else:
            try:
                text = Path(arguments.cb_table).read_text()
            except OSError as error:
                raise CbTableError(f"cannot read {arguments.cb_table}: {error}")
            settings, descriptions = parse_cb_table(text, count)
        header_text = render_header(settings, descriptions, count)
        Path(arguments.output).write_text(header_text)
    except CbTableError as error:
        print(f"{TOOL_NAME}: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"{TOOL_NAME}: {error}", file=sys.stderr)
        return 2
    print(f"{TOOL_NAME}: {summarize(descriptions, arguments.uniform_default, count)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
