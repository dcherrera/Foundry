# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Ingot: exact (lossless) model file compression. Build products go to ingot/build/.
#   make -C ingot          CLI + tests
#   make -C ingot test     run the test suites
CC ?= clang
ROOT := ..
B := build
CFLAGS := -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Wshadow -Wstrict-prototypes \
          -Wformat=2 -I$(ROOT) -D_DARWIN_C_SOURCE -D_DEFAULT_SOURCE
LDLIBS := -lm -lpthread

CORE_SRC := core/ingot_error.c core/ingot_hash.c core/ingot_parallel.c core/ingot_rans.c \
            core/ingot_input.c core/ingot_ggml.c $(ROOT)/tt/convert/ft_sha256.c
CODEC_SRC := codec/ingot_block.c codec/ingot_plain.c codec/ingot_delta.c codec/ingot_fcoder.c \
             codec/ingot_qtensor.c codec/ingot_q8.c codec/ingot_kquant.c \
             codec/ingot_kquant_fields.c \
             codec/ingot_blockdelta.c codec/ingot_predict.c codec/ingot_predict_plan.c \
             codec/ingot_match.c codec/ingot_match_plan.c codec/ingot_match_search.c \
             codec/ingot_tensor_codec.c
CONTAINER_SRC := container/ingot_header.c container/ingot_layout.c container/ingot_writer.c \
                 container/ingot_writer_tensor.c container/ingot_predict_rules.c \
                 container/ingot_writer_match.c container/ingot_match_pools.c \
                 container/ingot_reader.c container/ingot_stream.c container/ingot_base.c \
                 container/ingot_dirtail.c \
                 $(ROOT)/safetensors/safetensors.c
LIB_SRC := $(CORE_SRC) $(CODEC_SRC) $(CONTAINER_SRC)
HEADERS := $(wildcard core/*.h codec/*.h container/*.h)
TESTS := $(B)/test_ingot_rans $(B)/test_ingot_codec $(B)/test_ingot_container $(B)/test_ingot_red \
         $(B)/test_ingot_quant $(B)/test_ingot_stream $(B)/test_ingot_kquant \
         $(B)/test_ingot_predict $(B)/test_ingot_predict_O0 $(B)/test_ingot_match \
         $(B)/test_ingot_match_O0 $(B)/test_ingot_dict

.PHONY: all test clean bench red
bench: $(B)/bench_ingot_qtensor
all: $(B)/ingot $(TESTS)

$(B):
	mkdir -p $(B)

$(B)/ingot: cli/ingot_main.c cli/ingot_cli_info.c $(LIB_SRC) $(HEADERS) | $(B)
	$(CC) $(CFLAGS) -o $@ cli/ingot_main.c cli/ingot_cli_info.c $(LIB_SRC) $(LDLIBS)

$(B)/test_ingot_%: tests/test_ingot_%.c tests/ingot_test_files.h $(LIB_SRC) $(HEADERS) | $(B)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC) $(LDLIBS)

# The predict suite's integer-predictor tests again at -O0: contexts must not depend on the
# optimizer (both builds compare against the same golden hash).
$(B)/test_ingot_predict_O0: tests/test_ingot_predict.c tests/ingot_test_files.h $(LIB_SRC) \
                            $(HEADERS) | $(B)
	$(CC) $(subst -O2,-O0,$(CFLAGS)) -o $@ $< $(LIB_SRC) $(LDLIBS)

$(B)/test_ingot_match_O0: tests/test_ingot_match.c tests/ingot_test_files.h $(LIB_SRC) \
                          $(HEADERS) | $(B)
	$(CC) $(subst -O2,-O0,$(CFLAGS)) -o $@ $< $(LIB_SRC) $(LDLIBS)

# Red build: the dictionary suite without its planted rows must fail (proves its size check can).
$(B)/test_ingot_dict_red: tests/test_ingot_dict.c tests/ingot_test_files.h $(LIB_SRC) \
                          $(HEADERS) | $(B)
	$(CC) $(CFLAGS) -DINGOT_TEST_RED -o $@ $< $(LIB_SRC) $(LDLIBS)

red: $(B)/test_ingot_dict_red
	! ./$(B)/test_ingot_dict_red > $(B)/red.log 2>&1
	@echo "red: the unplanted dictionary suite fails, as it must"

# Coder benchmark on a real GGUF (not part of all / test): build/bench_ingot_qtensor FILE.gguf
$(B)/bench_ingot_qtensor: tests/bench_ingot_qtensor.c $(LIB_SRC) $(HEADERS) | $(B)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC) $(LDLIBS)

test: $(TESTS) $(B)/ingot
	./$(B)/test_ingot_rans
	./$(B)/test_ingot_codec
	./$(B)/test_ingot_container
	./$(B)/test_ingot_red
	./$(B)/test_ingot_quant
	./$(B)/test_ingot_stream
	./$(B)/test_ingot_kquant
	./$(B)/test_ingot_predict
	./$(B)/test_ingot_predict_O0 contexts
	./$(B)/test_ingot_match
	./$(B)/test_ingot_match_O0 contexts
	./$(B)/test_ingot_dict
	./tests/test_ingot_cli.sh
	$(MAKE) red

clean:
	rm -rf $(B)
