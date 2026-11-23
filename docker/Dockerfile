# Rocky Linux 9 (glibc 2.34) build container for chug -> bly (RHEL 9.9, 4x RTX A4500 / sm_86).
# Produces RHEL 9.9 ABI-compatible binaries with cuda-oxide (Rust 1.99.1 + LLVM 21 + CUDA 13.2 + NCCL)
# so nothing needs to be installed on bly beyond its existing NVIDIA driver (libcuda.so.1).

FROM nvidia/cuda:13.2.1-devel-rockylinux9

ARG RUST_TOOLCHAIN=1.99.1
ARG CUDA_RUST_REPO=https://github.com/NVIDIA/cuda-rust.git
ARG CUDA_RUST_REV=754b6d10e26919dfb2689d4e1d33da56ad039918

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN dnf -y install dnf-plugins-core epel-release \
    && dnf config-manager --set-enabled crb \
    && dnf -y install --allowerasing \
        ca-certificates \
        findutils \
        gcc \
        gcc-c++ \
        git \
        libffi-devel \
        libnccl \
        libnccl-devel \
        make \
        pkgconf-pkg-config \
        which \
        xz \
        zlib-devel \
        llvm21 \
        llvm21-devel \
        clang21 \
        clang21-devel \
        clang21-libs \
        lld21 \
    && dnf clean all

# Locate LLVM 21 bin/lib dirs (EPEL places llvm21 under /usr/lib64/llvm21) and expose symlinks.
RUN LLVM21_BIN="$(dirname "$(command -v llc-21 || find /usr -name 'llc*' -path '*llvm21*' -type f 2>/dev/null | head -n1)")" \
    && LLVM21_LIB="$(dirname "$(find /usr -name 'libclang.so*' -path '*llvm21*' 2>/dev/null | head -n1)")" \
    && ln -sfn "$LLVM21_BIN" /opt/llvm21-bin \
    && ln -sfn "$LLVM21_LIB" /opt/llvm21-lib

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDA_OXIDE_LLC=/opt/llvm21-bin/llc \
    LIBCLANG_PATH=/opt/llvm21-lib \
    LLVM_CONFIG_PATH=/opt/llvm21-bin/llvm-config \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    RUSTC_BOOTSTRAP=1 \
    PATH=/opt/cargo/bin:/opt/llvm21-bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install pinned Rust toolchain + required cuda-oxide components.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rustup component add --toolchain "${RUST_TOOLCHAIN}" \
        rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools

# Install cargo-oxide and pre-build the matching rustc-codegen-cuda backend at the pinned revision.
RUN git clone --filter=blob:none "${CUDA_RUST_REPO}" /tmp/cuda-rust \
    && git -C /tmp/cuda-rust checkout "${CUDA_RUST_REV}" \
    && cargo +"${RUST_TOOLCHAIN}" install --locked --path /tmp/cuda-rust/cuda-oxide/crates/cargo-oxide \
    && cargo +"${RUST_TOOLCHAIN}" build --release \
        --manifest-path /tmp/cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml \
    && mkdir -p /opt/cuda-oxide \
    && cp /tmp/cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so /opt/cuda-oxide/librustc_codegen_cuda.so \
    && rm -rf /tmp/cuda-rust /opt/cargo/registry /opt/cargo/git \
    && chmod -R a+rX /opt/rustup /opt/cargo /opt/cuda-oxide

ENV CUDA_OXIDE_BACKEND=/opt/cuda-oxide/librustc_codegen_cuda.so

# Collect runtime shared libraries to bundle alongside the chug binary on bly
# (bly already provides libcuda.so.1 via the NVIDIA driver).
RUN mkdir -p /opt/bly-runtime-libs \
    && for pattern in \
        "/usr/local/cuda/lib64/libcublas.so*" \
        "/usr/local/cuda/lib64/libcublasLt.so*" \
        "/usr/local/cuda/lib64/libcurand.so*" \
        "/usr/lib64/libnccl.so*" \
    ; do \
        for f in $pattern; do \
            [ -e "$f" ] && cp -a "$f" /opt/bly-runtime-libs/; \
        done; \
    done \
    && chmod -R a+rX /opt/bly-runtime-libs

WORKDIR /work
CMD ["/bin/bash"]
