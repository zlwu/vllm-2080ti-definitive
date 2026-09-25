#!/usr/bin/env bash
# Container entrypoint for the standalone vLLM 2080 Ti Definitive runtime.
#
# This image runs vLLM directly: there is no model-swapping proxy in front of it.
# The script reuses the upstream launcher's configuration resolution (profiles,
# startup mode, SM75 runtime environment, speculative decoding, CUDA graph
# sizing) and then `exec`s the server, so PID 1 is vLLM itself and `docker stop`
# reaches it as a normal SIGTERM.
#
# It mirrors launcher.sh's `run_start_flow`:
#   collect_config_env -> apply_mode -> set_sm75_runtime_env -> launch_server
# and skips only the parts that belong to a human-facing service manager (menu,
# pid file, log file, readiness polling, smoke test).
#
# Defaults live in the image's CMD; override them by passing arguments:
#   docker run ... <image> --model-dir /models/other --port 9000

set -euo pipefail

RUNTIME_TREE=${RUNTIME_TREE:-/opt/vllm-2080ti}
cd "$RUNTIME_TREE"

# shellcheck source=/dev/null
# Sourcing is safe: launcher.sh only calls main() when executed directly.
source "$RUNTIME_TREE/launcher.sh"

parse_launcher_args "$@"
register_env_config_overrides
apply_launcher_path_defaults
collect_config_env
apply_mode
set_sm75_runtime_env
print_review
configure_dflash_download_route
check_checkpoint_mmap_policy

# launch_server() injects this into the child env; we exec directly instead.
if [[ -n "${HF_ACTIVE_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="$HF_ACTIVE_ENDPOINT"
fi

# `--service-scope lan` makes the launcher bind 0.0.0.0 so the container can be
# reached directly at <host>:<port>; local scope would bind loopback only and
# would only work behind an in-container reverse proxy.
build_args "${VLLM_BIND_HOST:-0.0.0.0}"

printf -v args_text '%q ' "${VLLM_ARGS[@]}"
{
  echo "----------------------------------------------------------------"
  echo "vLLM 2080 Ti Definitive container start: $(date '+%F %T %Z')"
  echo "runtime tree:  $RUNTIME_TREE"
  echo "model:         $MODEL_DIR"
  echo "draft model:   ${SPECULATIVE_MODEL:-none}"
  echo "profile:       ${PROFILE:-manual}   mode: $MODE"
  echo "served name:   $SERVED_NAME"
  echo "bind:          ${VLLM_BIND_HOST:-0.0.0.0}:$PORT"
  echo "command:       $RUNTIME_TREE/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
  echo "----------------------------------------------------------------"
} >&2

exec "$RUNTIME_TREE/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
