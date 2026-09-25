# vllm-2080ti-definitive

A standalone container image for the **vLLM 2080 Ti Definitive Edition** (SM75)
runtime on dual RTX 2080 Ti 22 GB. One container, one job: it contains only the
inference runtime and serves vLLM's OpenAI-compatible API directly on a fixed
port. No model-swapping proxy and no llama.cpp are bundled.

## Security boundary

This public repository and its GHCR image contain only generic code and
placeholders. Do not commit real hostnames, IP addresses, domains, production
stack files, credentials, `.env` files, topology documents, or model weights.
Model weights and runtime caches are mounted at run time.

## Why this has to be built from source

There is **no published image** for the upstream runtime: the upstream
repository has no CI workflows, every release ships zero binary assets, and its
documentation contains no `docker pull` instructions. The runtime is also a
hardware-specific fork — SM75 CUDA/C++ kernels, a FlashQLA SM70/SM75
Gated-DeltaNet extension and a Rust front end — so `pip install vllm` cannot
reproduce it.

The build also has to happen on Ubuntu 24.04 (glibc 2.39), which is what
upstream's own Dockerfile targets as its final base: a runtime compiled on a
newer glibc cannot be loaded there. That is why stage 1 pins an Ubuntu 24.04
builder rather than using whatever the host runs.

## What is in the image

| Stage | Content |
|---|---|
| `runtime-builder` → `runtime-builder-src` | `nvidia/cuda:13.0.3-devel-ubuntu24.04` + GCC 15 + Rust 1.95 + uv, running the upstream `build.sh` into `/opt/vllm-2080ti` |
| `runtime` (final) | `nvidia/cuda:13.0.3-base-ubuntu24.04` + that runtime tree + a trimmed `/usr/local/cuda-13.0` (nvcc, nvvm, headers, libcudart) for run-time kernel JIT |

Build leftovers are pruned inside the builder stage before the copy — the Rust
build tree, the CUDA static/math libraries that torch already ships under
`.venv/.../nvidia`, and the `triton_kernels` FetchContent source — which cuts the
image roughly in half. Pruning after the copy would only add whiteouts.

## When the image is rebuilt

| Trigger | Effect |
|---|---|
| Pinned CUDA base digest moves (weekly check) | Builds, then records the new digests in `.github/cuda-base` |
| New stable release of `weicj/vLLM-2080Ti-Definitive` | Opens a pull request bumping `.github/vllm-ref`; merging it builds |
| `Dockerfile` / `entrypoint.sh` / pin files change | Builds |
| `workflow_dispatch` | Always builds |
| Pull request | Validation only: `shellcheck` + `docker build --target toolchain` |

A new upstream vLLM release goes through review because it changes runtime
behaviour. Note that **profile filenames have changed between upstream
releases** (`dflash2-fp8kv-1x256k-…` at v0.2.1 vs `dflash2-fp8kv-1x262K-…` at
v0.2.2); a stale name makes the launcher silently fall back to defaults.

## Usage

```bash
docker run -d --name vllm \
  --gpus all \
  --shm-size=2g \
  -p 8089:8089 \
  -v /path/to/models:/models \
  -v /path/to/cache:/data/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/data/vllm/torchinductor \
  -e TRITON_CACHE_DIR=/data/vllm/triton \
  -e FLASHINFER_WORKSPACE_BASE=/data/vllm/flashinfer \
  -e VLLM_CACHE_ROOT=/data/vllm/vllm-cache \
  ghcr.io/zlwu/vllm-2080ti-definitive:latest
```

The image's `CMD` carries a sensible default (model paths, profile, TP=2, port
8089, `--service-scope lan` so vLLM binds `0.0.0.0`). Override it by passing
arguments — they go straight to the upstream launcher:

```bash
docker run ... ghcr.io/zlwu/vllm-2080ti-definitive:latest \
  --model-dir /models/vllm/OtherCheckpoint --port 9000
```

The expected layout under the mount is `/models/vllm/<checkpoint>`, and a DFlash2
draft checkpoint is optional (`--speculative-model`).

### Runtime requirements

* **`/dev/shm` must be larger than Docker's 64 MiB default.** Tensor-parallel
  bootstrapping needs ~160 MiB and fails with `Insufficient space in /dev/shm`.
  Use `--shm-size=2g`; under Swarm, mount a tmpfs at `/dev/shm` instead (services
  have no `--shm-size`).
* **On an overlay network, NCCL needs to be pointed at loopback.** NCCL picks the
  container's overlay `eth0` for the TP bootstrap and then times out connecting to
  the container's own overlay IP (`ncclCommInitRank` never returns). Set
  `NCCL_SOCKET_IFNAME=lo` and `NCCL_IB_DISABLE=1`. With `--network host` this is
  not needed.
* **`vm.overcommit_memory=1` on the host.** The upstream launcher refuses to start
  when the largest `safetensors` file exceeds the commit headroom, and a container
  cannot change a host sysctl.
* **A compiler at run time** is why the image carries GCC 15: FlashInfer compiles
  SM75 kernels and `torch.compile` generates C++ wrappers on first use. Point
  them at a persistent mount so a container restart does not recompile.

### Swarm example

```yaml
services:
  vllm:
    image: ghcr.io/zlwu/vllm-2080ti-definitive:latest
    volumes:
      - /path/to/models:/models
      - /path/to/cache:/data/vllm
      - type: tmpfs
        target: /dev/shm
        tmpfs:
          size: 2147483648
    ports:
      - target: 8089
        published: 8089
        mode: host
    environment:
      - NCCL_SOCKET_IFNAME=lo
      - NCCL_IB_DISABLE=1
      - TORCHINDUCTOR_CACHE_DIR=/data/vllm/torchinductor
      - TRITON_CACHE_DIR=/data/vllm/triton
      - FLASHINFER_WORKSPACE_BASE=/data/vllm/flashinfer
      - VLLM_CACHE_ROOT=/data/vllm/vllm-cache
    deploy:
      placement:
        constraints:
          - node.labels.gpu == "true"
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Building locally

```bash
docker build -t vllm-2080ti-definitive:local \
  --build-arg VLLM_REF="$(head -1 .github/vllm-ref)" \
  --build-arg MAX_JOBS=4 .
```

Compiles vLLM from source; takes tens of minutes on 4 cores. To validate only
the builder prerequisites in a few minutes:

```bash
docker build --target toolchain -t vllm-2080ti-definitive:toolchain .
```

## Attribution and license

This image packages and redistributes the work of others:

* [vLLM](https://github.com/vllm-project/vllm) — Apache-2.0.
* [vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)
  by [github.com/weicj](https://github.com/weicj) — the SM75 fork, launcher,
  profiles and validation evidence this image builds.
* [FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75) — SM70/SM75
  Gated-DeltaNet prefill backend.

Everything here is licensed under Apache-2.0; see [LICENSE](LICENSE). This
repository is an independent packaging project, not affiliated with or endorsed
by the upstream authors. Route parameters, performance figures and support
statements belong to the upstream project — consult its documentation for the
authoritative list of validated profiles.
