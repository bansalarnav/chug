#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional local overrides (gitignored), e.g. BLY_REMOTE_DIR, BLY_GPUS, CHUG_ARCH.
if [[ -f "$REPO_ROOT/.env.bly" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env.bly"
fi

IMAGE_NAME="${CHUG_BUILDER_IMAGE:-chug-builder:rocky9-cuda13.2}"
BLY_HOST="${BLY_HOST:-bly}"
BLY_REMOTE_DIR="${BLY_REMOTE_DIR:-~/chug-run}"
BLY_GPUS="${BLY_GPUS:-0,1,2,3}"
CHUG_ARCH="${CHUG_ARCH:-sm_86}"
CHUG_BIN_NAME="${CHUG_BIN_NAME:-chug}"
CHUG_MATERIALIZE_CUBIN="${CHUG_MATERIALIZE_CUBIN:-0}"
CACHE_DIR="${CHUG_CACHE_DIR:-$HOME/.cache/chug-builder}"
DIST_DIR="$REPO_ROOT/.bly-dist"
SSH_CTL="/tmp/chug-bly-ssh-%C"

SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o "ControlPath=$SSH_CTL"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args...]

Commands:
  init                   Build the local Rocky Linux 9 cuda-oxide container & extract runtime .so libs
  doctor                 Run 'cargo oxide doctor' inside the builder container
  build [oxide args...]  Compile chug for ${CHUG_ARCH} (glibc 2.34) inside the container
  sync                   Rsync built binary + runtime .so libs to ${BLY_HOST}:${BLY_REMOTE_DIR}
  run [-- app args...]   Build locally, sync to ${BLY_HOST}, and run on its GPUs
  exec <cmd...>          Run an arbitrary command inside the builder container
  shell                  Open an interactive shell inside the builder container
  clean-remote           Remove ${BLY_REMOTE_DIR} on ${BLY_HOST}

Environment overrides (or put them in .env.bly):
  BLY_HOST=${BLY_HOST}
  BLY_REMOTE_DIR=${BLY_REMOTE_DIR}
  BLY_GPUS=${BLY_GPUS}
  CHUG_ARCH=${CHUG_ARCH}
  CHUG_MATERIALIZE_CUBIN=${CHUG_MATERIALIZE_CUBIN}
EOF
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    echo "Error: cannot talk to Docker daemon as $(whoami)." >&2
    echo "Add yourself to the docker group once:" >&2
    echo "  sudo usermod -aG docker \$USER && newgrp docker" >&2
    exit 1
  fi
}

ensure_image() {
  if ! docker_cmd image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Builder image '$IMAGE_NAME' not found. Running init..." >&2
    cmd_init
  fi
}

run_in_container() {
  local tty_flags=()
  if [[ -t 0 && -t 1 ]]; then
    tty_flags=(-it)
  fi
  mkdir -p "$CACHE_DIR/cargo" "$CACHE_DIR/target"
  docker_cmd run --rm --entrypoint "" "${tty_flags[@]}" \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT:/work" \
    -v "$CACHE_DIR/cargo:/cache/cargo" \
    -v "$CACHE_DIR/target:/cache/target" \
    -w /work \
    -e CARGO_HOME=/cache/cargo \
    -e CARGO_TARGET_DIR=/cache/target \
    -e HOME=/tmp \
    -e CUDA_OXIDE_TARGET="$CHUG_ARCH" \
    "$IMAGE_NAME" \
    "$@"
}

extract_libs() {
  mkdir -p "$DIST_DIR/libs"
  docker_cmd run --rm --entrypoint "" \
    --user "$(id -u):$(id -g)" \
    -v "$DIST_DIR/libs:/out" \
    "$IMAGE_NAME" \
    bash -euo pipefail -c 'cp -a /opt/bly-runtime-libs/. /out/'
}

cmd_init() {
  echo "==> Building Rocky Linux 9 builder image ($IMAGE_NAME)..."
  docker_cmd build -t "$IMAGE_NAME" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT/docker"
  echo "==> Extracting bundled CUDA/NCCL runtime libraries..."
  extract_libs
  echo "==> Done. Runtime libs saved to $DIST_DIR/libs"
}

cmd_doctor() {
  ensure_image
  run_in_container cargo oxide doctor
}

cmd_build() {
  ensure_image
  local oxide_flags=()
  if [[ "$CHUG_MATERIALIZE_CUBIN" == "1" ]]; then
    oxide_flags+=(--materialize-cubin)
  fi
  echo "==> Building $CHUG_BIN_NAME for $CHUG_ARCH (glibc 2.34)..."
  run_in_container cargo oxide "${oxide_flags[@]}" build --arch "$CHUG_ARCH" "$@"

  mkdir -p "$DIST_DIR/bin"
  if [[ ! -f "$CACHE_DIR/target/release/$CHUG_BIN_NAME" ]]; then
    echo "Error: expected binary at $CACHE_DIR/target/release/$CHUG_BIN_NAME" >&2
    exit 1
  fi
  cp -f "$CACHE_DIR/target/release/$CHUG_BIN_NAME" "$DIST_DIR/bin/$CHUG_BIN_NAME"
  if [[ ! -d "$DIST_DIR/libs" || -z "$(ls -A "$DIST_DIR/libs" 2>/dev/null)" ]]; then
    extract_libs
  fi
  echo "==> Built: $DIST_DIR/bin/$CHUG_BIN_NAME"
}

cmd_sync() {
  if [[ ! -f "$DIST_DIR/bin/$CHUG_BIN_NAME" ]]; then
    echo "Error: binary not built yet. Run '$0 build' first." >&2
    exit 1
  fi
  echo "==> Syncing binary + runtime libs to ${BLY_HOST}:${BLY_REMOTE_DIR}..."
  ssh "${SSH_OPTS[@]}" "$BLY_HOST" "mkdir -p ${BLY_REMOTE_DIR}/bin ${BLY_REMOTE_DIR}/libs"
  rsync -az --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    "$DIST_DIR/bin/" "${BLY_HOST}:${BLY_REMOTE_DIR}/bin/"
  rsync -az \
    -e "ssh ${SSH_OPTS[*]}" \
    "$DIST_DIR/libs/" "${BLY_HOST}:${BLY_REMOTE_DIR}/libs/"
}

cmd_run() {
  local app_args=()
  if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
  fi
  app_args=("$@")

  cmd_build
  cmd_sync

  local quoted_args=""
  if [[ ${#app_args[@]} -gt 0 ]]; then
    quoted_args="$(printf ' %q' "${app_args[@]}")"
  fi

  echo "==> Running on ${BLY_HOST} (GPUs: ${BLY_GPUS})..."
  ssh "${SSH_OPTS[@]}" -t "$BLY_HOST" \
    "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${BLY_GPUS} LD_LIBRARY_PATH=${BLY_REMOTE_DIR}/libs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; exec ${BLY_REMOTE_DIR}/bin/${CHUG_BIN_NAME}${quoted_args}"
}

cmd_exec() {
  ensure_image
  run_in_container "$@"
}

cmd_shell() {
  ensure_image
  run_in_container /bin/bash
}

cmd_clean_remote() {
  echo "==> Removing ${BLY_HOST}:${BLY_REMOTE_DIR}..."
  ssh "${SSH_OPTS[@]}" "$BLY_HOST" "rm -rf ${BLY_REMOTE_DIR}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  local cmd="$1"
  shift
  case "$cmd" in
    init) cmd_init "$@" ;;
    doctor) cmd_doctor "$@" ;;
    build) cmd_build "$@" ;;
    sync) cmd_sync "$@" ;;
    run) cmd_run "$@" ;;
    exec) cmd_exec "$@" ;;
    shell) cmd_shell "$@" ;;
    clean-remote) cmd_clean_remote "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
