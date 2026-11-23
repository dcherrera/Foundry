# Reproducible cross-build of the K80 BAR1-strap kernel for the K80 host.
#
# Produces an x86-64 Ubuntu linux-image .deb (kernel 6.8.0-106.106 + the
# built-in GK210 PSTRAPS pci_fixup_early) by CROSS-COMPILING — so it builds at
# native speed on an arm64 host (Apple Silicon) with no emulation.
#
# Reproducibility: the kernel source is pinned to an exact Ubuntu version (pulled
# from Launchpad, which keeps every version permanently) and the config is the
# target host's own /boot/config, fetched at build time by build.sh (never committed).  Same inputs -> same kernel.  For
# bit-exact rebuilds also pin the base image by digest and point apt at
# snapshot.ubuntu.com.
#
# Layer order keeps the big source fetch cached independently of the build-deps
# layer, so adding a build dependency does not re-download the kernel source.
#
# Build:  see build.sh (wraps `docker build` + copies the .deb out).

FROM ubuntu:24.04

ARG KSRC_VERSION=6.8.0-106.106
ENV DEBIAN_FRONTEND=noninteractive \
    ARCH=x86_64 \
    CROSS_COMPILE=x86_64-linux-gnu- \
    KSRC_VERSION=${KSRC_VERSION}

WORKDIR /build

# 1) Tools needed only to fetch the exact historical kernel source from Launchpad.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ubuntu-dev-tools ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2) Fetch + extract the exact Ubuntu kernel source (extracts to linux-*/).
#    Cached unless KSRC_VERSION changes.
RUN pull-lp-source linux ${KSRC_VERSION}

# 3) Build dependencies: native host compiler (HOSTCC) AND the x86-64 cross
#    toolchain (target CC), plus kernel build + Debian packaging tools.  zstd is
#    required because the target config sets CONFIG_KERNEL_ZSTD / MODULE_COMPRESS_ZSTD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-x86-64-linux-gnu g++-x86-64-linux-gnu \
        bc bison flex libssl-dev libelf-dev \
        kmod cpio rsync xz-utils zstd dpkg-dev debhelper fakeroot && \
    rm -rf /var/lib/apt/lists/*

# 4) Drop in the built-in quirk and the target's config.
COPY k80-strap-quirk.c /build/k80-strap-quirk.c
COPY target.config /build/target.config

# 5) Patch the tree, configure from the target's config, cross-build the .deb.
#    Neutralise Ubuntu's Canonical signing-key paths and disable module signing
#    (Secure Boot is off on the target).  LOCALVERSION forks the kernel to
#    6.8.0-k80strap so it coexists with the stock 6.8.0-106-generic.
# (rm -rf debian: drop Ubuntu's heavy packaging so upstream bindeb-pkg generates
#  its own minimal package with only the deps we installed.)
RUN set -eux; \
    cd "$(echo linux-*/ | head -1)"; \
    rm -rf debian; \
    cp /build/k80-strap-quirk.c drivers/pci/k80-strap-quirk.c; \
    grep -q 'k80-strap-quirk.o' drivers/pci/Makefile || \
        echo 'obj-y += k80-strap-quirk.o' >> drivers/pci/Makefile; \
    cp /build/target.config .config; \
    ./scripts/config --set-str SYSTEM_TRUSTED_KEYS ""; \
    ./scripts/config --set-str SYSTEM_REVOCATION_KEYS ""; \
    ./scripts/config --disable MODULE_SIG; \
    ./scripts/config --disable DEBUG_INFO_BTF; \
    ./scripts/config --disable LOCALVERSION_AUTO; \
    ./scripts/config --set-str LOCALVERSION "-k80strap"; \
    make ARCH="$ARCH" CROSS_COMPILE="$CROSS_COMPILE" olddefconfig; \
    make ARCH="$ARCH" CROSS_COMPILE="$CROSS_COMPILE" -j"$(nproc)" bindeb-pkg; \
    mkdir -p /out; cp /build/linux-image-*.deb /out/

# The .deb lands in /out; build.sh copies it to the host.
