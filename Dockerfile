# syntax=docker/dockerfile:1.7
#
# Standalone vLLM 2080 Ti Definitive Edition runtime for SM75 (dual RTX 2080 Ti).
#
# One container, one job: this image contains only the inference runtime. It does
# not bundle a model-swapping proxy, and it does not bundle llama.cpp. It starts
# vLLM directly and serves the OpenAI-compatible API on a fixed port.
#
# Stage 1 compiles the upstream SM75 runtime from source. It must run on Ubuntu
# 24.04 (glibc 2.39) because stage 2 is Ubuntu 24.04 based — that is also the
# platform upstream's own Dockerfile targets as its final base — and a runtime
# built against a newer glibc cannot be loaded there.
#
# Build (hours on 4 cores):
#   docker build -t vllm-2080ti-definitive:local .
#
# Validate only the builder prerequisites (minutes):
#   docker build --target toolchain -t vllm-2080ti-definitive:toolchain .

ARG CUDA_DEVEL_BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG CUDA_RUNTIME_BASE=nvidia/cuda:13.0.3-base-ubuntu24.04


# --------------------------------------------------------------------------
# Stage 1: compile the SM75 vLLM runtime
# --------------------------------------------------------------------------
FROM ${CUDA_DEVEL_BASE} AS runtime-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REF=v0.2.2
ARG REPO_URL=https://github.com/weicj/vLLM-2080Ti-Definitive.git
# Roughly (RAM_GiB - 3) / 3, capped by the CPU count. nvcc front-end jobs are
# memory hungry; raise it only on a machine that has the RAM.
ARG MAX_JOBS=4
# build.sh only pre-installs torch when its network preflight selects a domestic
# PyPI mirror. On the official route it relies on `uv pip install
# --torch-backend`, but build isolation is disabled, so the build backend's own
# torch requirement goes unmet and the build dies with "No module named
# 'torch'". Pin the PyTorch index so torch is installed first on any route.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG RUSTUP_DIST_SERVER=
ARG RUSTUP_UPDATE_ROOT=
ARG RUST_TOOLCHAIN=1.95

ENV RUNTIME_TREE=/opt/vllm-2080ti \
    TORCH_CUDA_ARCH_LIST=7.5 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin

# GCC 15 is what the 0.2.x line validates; Ubuntu 24.04 only ships gcc-13/14.
# Symlinks in /usr/local/bin win over /usr/bin on PATH, so `gcc -dumpversion`
# (used by the upstream build script) and nvcc's default host compiler both
# resolve to 15.
#
# `apt_install` retries the update+install pair together. The Ubuntu archive
# rotates packages while a CI job runs, so a freshly fetched index can already
# point at a pool file that 404s seconds later; retrying the update alone is not
# enough, and the whole build would fail on a hiccup that clears in seconds.
RUN set -eux; \
    apt_install() { \
      for i in 1 2 3 4 5; do \
        if apt-get update -o Acquire::Retries=5; then \
          if apt-get install -y --no-install-recommends -o Acquire::Retries=5 "$@"; then return 0; fi; \
        fi; \
        echo "apt attempt $i failed; retrying in 20s"; sleep 20; \
      done; return 1; }; \
    apt_install software-properties-common gnupg ca-certificates curl git make pkg-config perl; \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
    apt_install gcc-15 g++-15 \
      python3.12 python3.12-venv python3.12-dev \
      ninja-build libnuma-dev; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/gcc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/g++; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/cc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/c++; \
    rm -rf /var/lib/apt/lists/*; \
    gcc -dumpversion; \
    python3.12 --version

# The upstream runtime builds Rust artifacts (a `vllm-rs` binary and a PyO3
# parser module) through setuptools-rust, so cargo must exist before build.sh.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain none --no-modify-path \
 && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal \
 && rustup default "${RUST_TOOLCHAIN}" \
 && cargo --version \
 && rustc --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
 && install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx \
 && uv --version


# --------------------------------------------------------------------------
# Stage 1a: toolchain only, so prerequisite changes fail fast
# --------------------------------------------------------------------------
FROM runtime-builder AS toolchain

RUN gcc -dumpversion && g++ -dumpversion && cargo --version && uv --version


# --------------------------------------------------------------------------
# Stage 1b: source build
# --------------------------------------------------------------------------
FROM toolchain AS runtime-builder-src

# A shallow clone made from a release tag still carries that tag, so
# setuptools_scm can derive the version.
RUN git clone --depth 1 --branch "${VLLM_REF}" "${REPO_URL}" "${RUNTIME_TREE}" \
 && git -C "${RUNTIME_TREE}" describe --tags --exact-match

# Upstream's FlashQLA loader refuses to build unless torch.cuda.is_available(),
# even though the extension is compiled by nvcc from TORCH_CUDA_ARCH_LIST and
# never touches a device. Build hosts and CI runners have no GPU, so relax
# exactly that guard behind an explicit opt-in that is only set for the build.
COPY relax-flashqla-build-guard.py /tmp/relax-flashqla-build-guard.py
RUN python3 /tmp/relax-flashqla-build-guard.py \
      "${RUNTIME_TREE}/tools/flashqla_sm75_patches/sm_legacy.py"

# Keep the two host gates that actually shape the artifacts, since
# ALLOW_HOST_MISMATCH below relaxes build.sh's own checks wholesale.
RUN set -eux; \
    nvcc --version | grep -q 'release 13\.'; \
    test "$(gcc -dumpversion | cut -d. -f1)" = "15"; \
    echo "toolchain preflight ok: $(nvcc --version | tail -1)"; \
    echo "python: $(python3.12 --version)"; \
    echo "build kernel: $(uname -r)"

# The long step. build.sh runs its own host checks, picks PyPI/Git mirrors,
# creates the venv, compiles vLLM, patches torch inductor for E8M0, fetches and
# builds the FlashQLA SM70/SM75 extension, and finally validates the runtime.
#
# ALLOW_HOST_MISMATCH waives build.sh's `kernel >= 7` requirement. That check
# describes the deployment host (the 0.2.x line is validated on Ubuntu 26.04 /
# kernel 7, which the target GPU node runs); this builder is Ubuntu 24.04 and
# the host kernel cannot influence the compiled artifacts. nvcc and GCC are
# asserted explicitly just above instead.
RUN cd "${RUNTIME_TREE}" \
 && ASSUME_YES=1 NON_INTERACTIVE=1 MAX_JOBS="${MAX_JOBS}" \
    BUILD_TORCH_INDEX="${TORCH_INDEX}" \
    FLASHQLA_ALLOW_GPU_LESS_BUILD=1 ALLOW_HOST_MISMATCH=1 ./build.sh

# Trim what serving does not need. Everything here runs in the builder stage
# before the COPY: deleting files in a later layer would only add whiteout
# entries and leave the bytes in the image.
#
#   rust/target            5.1 GB  built artifacts, already installed into vllm/
#   .deps/*-src           ~0.9 GB  FetchContent inputs. triton_kernels is not
#                                  importable from the venv (verified); cutlass
#                                  and flash_qla are kept because they are.
#   cuda-13.0 targets/    ~4.0 GB  static and math libraries. torch ships its own
#                                  CUDA 13 libs under .venv/.../nvidia, and this
#                                  toolkit exists only so nvcc can JIT: nvcc +
#                                  nvvm + headers + libcudart are what stay.
#   cuda-13.0 compat/     ~0.3 GB  driver compat shims, unused with a modern driver
#   __pycache__/*.pyc     ~60 MB   both smoke tests below run with
#                                  PYTHONDONTWRITEBYTECODE=1 so nothing reappears
#   cutlass_dsl cu12      ~215 MB  the package ships cu12 and cu13 builds of the
#                                  same MLIR runtime; this image is CUDA 13 only
#   nvidia/cu13 bin+nvvm  ~250 MB  a second copy of the CUDA compiler; CUDA_HOME
#                                  resolves to /usr/local/cuda-13.0
#   triton cupti          ~128 MB  CUPTI is profiling-only, never used to serve
RUN set -eux; \
    rm -rf "${RUNTIME_TREE}/rust/target" \
           "${RUNTIME_TREE}/.git" \
           "${RUNTIME_TREE}/build-logs" \
           "${RUNTIME_TREE}/.deps/triton_kernels-src" \
           "${RUNTIME_TREE}/.deps/triton_kernels-subbuild" \
           "${RUNTIME_TREE}/.deps/cutlass-build" \
           "${RUNTIME_TREE}/.deps/cutlass-subbuild"; \
    find "${RUNTIME_TREE}" -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    find "${RUNTIME_TREE}" -name '*.pyc' -delete; \
    find "${RUNTIME_TREE}" -maxdepth 3 -type d -name tests -prune -exec rm -rf {} +; \
    SITE_PACKAGES="$("${RUNTIME_TREE}/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"; \
    rm -rf "${SITE_PACKAGES}/nvidia_cutlass_dsl/cu12"; \
    find "${SITE_PACKAGES}/nvidia_cutlass_dsl" -name '*cu12*' -delete; \
    rm -rf "${SITE_PACKAGES}/nvidia/cu13/bin" "${SITE_PACKAGES}/nvidia/cu13/nvvm"; \
    find "${SITE_PACKAGES}" -type d -name cupti -prune -exec rm -rf {} +; \
    CUDA_ROOT=/usr/local/cuda-13.0; \
    rm -rf "${CUDA_ROOT}/compat" "${CUDA_ROOT}/compute-sanitizer" "${CUDA_ROOT}/extras" \
           "${CUDA_ROOT}/doc" "${CUDA_ROOT}/src" "${CUDA_ROOT}/gds" "${CUDA_ROOT}/nvml"; \
    find "${CUDA_ROOT}/targets" -name '*.a' -not -name 'libcudadevrt.a' -delete; \
    rm -f "${CUDA_ROOT}"/targets/*/lib/libcublas* "${CUDA_ROOT}"/targets/*/lib/libcufft* \
          "${CUDA_ROOT}"/targets/*/lib/libcusolver* "${CUDA_ROOT}"/targets/*/lib/libcusparse* \
          "${CUDA_ROOT}"/targets/*/lib/libcurand* "${CUDA_ROOT}"/targets/*/lib/libnpp* \
          "${CUDA_ROOT}"/targets/*/lib/libnvjpeg* "${CUDA_ROOT}"/targets/*/lib/libcufile* \
          "${CUDA_ROOT}"/targets/*/lib/libnvrtc* "${CUDA_ROOT}"/targets/*/lib/libnvblas*; \
    find "${CUDA_ROOT}" -xtype l -delete; \
    nvcc --version | tail -1; \
    du -sh "${CUDA_ROOT}" "${RUNTIME_TREE}"

# Strip debug symbols. Upstream's Rust and C++ artifacts ship unstripped: vllm-rs
# alone is 584 MB of which 81% is symbols, libtriton.so 439 MB -> 172 MB, and the
# NVIDIA pip libraries carry symbols too. `--strip-unneeded` keeps the dynamic
# symbol tables that JIT linking and dlopen need and only touches shared objects
# and executables, so behaviour is unchanged; the cost is unusable backtraces.
# No static archives remain by this point, so nothing link-time is damaged.
RUN set -eux; \
    before=$(du -sm "${RUNTIME_TREE}" | cut -f1); \
    find "${RUNTIME_TREE}" -type f \( -name '*.so' -o -name '*.so.*' \) -print0 \
      | xargs -0 -r -n 40 strip --strip-unneeded 2>/dev/null || true; \
    find "${RUNTIME_TREE}" /usr/local/cuda-13.0 -type f -executable -print0 \
      | xargs -0 -r -n 40 strip --strip-unneeded 2>/dev/null || true; \
    after=$(du -sm "${RUNTIME_TREE}" | cut -f1); \
    echo "strip: ${before} MB -> ${after} MB (saved $((before - after)) MB)"; \
    nvcc --version | tail -1

# PYTHONDONTWRITEBYTECODE matters: importing torch/vllm here would otherwise
# regenerate the __pycache__ the prune step just removed, and the COPY below
# would carry it into the image.
RUN PYTHONDONTWRITEBYTECODE=1 "${RUNTIME_TREE}/.venv/bin/python" -c \
      'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'


# --------------------------------------------------------------------------
# Stage 2: the serving image
# --------------------------------------------------------------------------
FROM ${CUDA_RUNTIME_BASE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG RUNTIME_TREE=/opt/vllm-2080ti

ENV RUNTIME_TREE=${RUNTIME_TREE} \
    CUDA_HOME=/usr/local/cuda-13.0 \
    CUDA_PATH=/usr/local/cuda-13.0 \
    TORCH_CUDA_ARCH_LIST=7.5 \
    FLASHINFER_ENABLE_AOT=1 \
    VLLM_DISABLE_TILELANG=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONSAFEPATH=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    VLLM_PORT=8089

# Same apt retry rationale as the builder stage.
#
# A compiler is needed at *runtime*: FlashInfer compiles SM75 kernels and
# torch.compile generates C++ wrappers on first use (then cached under
# FLASHINFER_WORKSPACE_BASE / TORCHINDUCTOR_CACHE_DIR). Triton additionally
# compiles a tiny `cuda_utils.c` on import, so the Python headers must be
# present as well.
RUN set -eux; \
    apt_install() { \
      for i in 1 2 3 4 5; do \
        if apt-get update -o Acquire::Retries=5; then \
          if apt-get install -y --no-install-recommends -o Acquire::Retries=5 "$@"; then return 0; fi; \
        fi; \
        echo "apt attempt $i failed; retrying in 20s"; sleep 20; \
      done; return 1; }; \
    apt_install software-properties-common gnupg ca-certificates curl; \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
    apt_install gcc-15 g++-15 libnuma1 libgomp1 python3.12 python3.12-dev; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/gcc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/g++; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/cc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/c++; \
    rm -rf /var/lib/apt/lists/*

# CUDA 13.0 toolchain for run-time JIT. nvcc + nvvm + headers + libcudart is all
# the runtime needs; the base image's own minimal CUDA is the same version, so
# this merge is consistent.
COPY --from=runtime-builder-src /usr/local/cuda-13.0 /usr/local/cuda-13.0
COPY --from=runtime-builder-src ${RUNTIME_TREE} ${RUNTIME_TREE}

COPY entrypoint.sh ${RUNTIME_TREE}/entrypoint.sh
RUN chmod 0755 ${RUNTIME_TREE}/entrypoint.sh

RUN PYTHONDONTWRITEBYTECODE=1 ${RUNTIME_TREE}/.venv/bin/python -c \
      'import torch, vllm; print("runtime ok:", vllm.__version__, torch.__version__, torch.version.cuda)'

# 8089 must match the default `--port` in CMD below. `--service-scope lan` is
# what makes vLLM bind 0.0.0.0 instead of loopback, so clients can reach the
# container directly at <host>:8089 without a reverse proxy in front.
EXPOSE 8089
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8089/health || exit 1

ENTRYPOINT ["/opt/vllm-2080ti/entrypoint.sh"]
CMD ["--model-dir", "/models/vllm/Qwen3.8-27B-NVFP4", \
     "--speculative-model", "/models/vllm/Qwen3.8-27B-DFlash2", \
     "--profile", "2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env", \
     "--mode", "fast", \
     "--gpu-devices", "0,1", \
     "--tp-size", "2", \
     "--pp-size", "1", \
     "--served-name", "qwen3.8-nvfp4", \
     "--service-scope", "lan", \
     "--port", "8089"]
