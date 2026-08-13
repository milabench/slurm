#!/bin/bash
# Prepare milabench + vLLM CPU deps on a Trillium CPU login node, then submit
# one compute job per diagnostic bench (TP/PP × 2/4 nodes). Compute nodes have
# no internet and cannot write $HOME/$PROJECT
# (see https://docs.alliancecan.ca/wiki/Trillium).
#
# Usage (on trillium.alliancecan.ca — CPU login, not GPU):
#   ./trilium_vllm_schedule.sh
#
# Default: submit 2-node TP and PP smokes. Override benches / sizes:
#   BENCHMARKS="vllm-opt-125m-fp32-tp-nodes vllm-opt-125m-fp32-tp-4nodes" ./trilium_vllm_schedule.sh
#   BENCHMARKS="vllm-opt-125m-fp32-pp-4nodes" NODES=4 ./trilium_vllm_schedule.sh
#
# Optional env overrides:
#   MILABENCH_SOURCE / MILABENCH_WORKDIR / ACCOUNT / TIME / PARTITION
#   PYTHON_VERSION / PYTORCH_VERSION / CPUS_PER_TASK / FOLLOW=0
#   SKIP_INSTALL=1  (reuse an already-prepared workspace)

set -euo pipefail

# Bump when changing install logic (grep err.txt for this string to confirm cluster copy).
SCHEDULE_SCRIPT_VERSION="2026-08-13-cpu-vllm-workaround-v4"
echo "trilium_vllm_schedule.sh ${SCHEDULE_SCRIPT_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/trilium_vllm_run.sbatch"

# --- defaults -----------------------------------------------------------------
export ACCOUNT="${ACCOUNT:-rrg-bengioy-ad}"
# vLLM CPU + Ray bring-up is slower than torchsrun; allow headroom.
export TIME="${TIME:-1:00:00}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
export CUDA_VERSION="${CUDA_VERSION:-130}"
export PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
export MILABENCH_GPU_ARCH="${MILABENCH_GPU_ARCH:-cpu}"
# Space-separated list of diagnostic benches (one Slurm job each).
export BENCHMARKS="${BENCHMARKS:-vllm-opt-125m-fp32-tp-nodes vllm-opt-125m-fp32-pp-nodes}"
# Trillium CPU nodes are whole 192-core nodes.
export CPUS_PER_TASK="${CPUS_PER_TASK:-192}"
export PARTITION="${PARTITION:-debug}"
# Tail the first submitted job's log (set FOLLOW=0 to exit after submit).
export FOLLOW="${FOLLOW:-1}"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: \$SCRATCH is unset. Run this on a Trillium login node." >&2
  exit 1
fi

# Scripts live in <repo>/slurm/; milabench is usually <repo>/milabench.
# On cluster some checkouts keep milabench next to the scripts — accept both.
if [[ -z "${MILABENCH_SOURCE:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../milabench/milabench" ]]; then
    MILABENCH_SOURCE="${SCRIPT_DIR}/../milabench"
  elif [[ -d "${SCRIPT_DIR}/milabench/milabench" ]]; then
    MILABENCH_SOURCE="${SCRIPT_DIR}/milabench"
  else
    MILABENCH_SOURCE="${SCRIPT_DIR}/../milabench"
  fi
fi
MILABENCH_SOURCE="$(cd "${MILABENCH_SOURCE}" && pwd)"
export MILABENCH_SOURCE
export MILABENCH_WORKDIR="${MILABENCH_WORKDIR:-${SCRATCH}/milabench_vllm_ray}"
export MILABENCH_ENV="${MILABENCH_ENV:-${MILABENCH_WORKDIR}/.env/${PYTHON_VERSION}}"
export MILABENCH_BASE="${MILABENCH_BASE:-${MILABENCH_WORKDIR}/results}"
export MILABENCH_CONFIG="${MILABENCH_CONFIG:-${MILABENCH_SOURCE}/config/diagnostic/diagnostic.yaml}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MILABENCH_WORKDIR}/uv-cache}"
export PYTHONUNBUFFERED=1
export MILABENCH_USE_TOML_DEPS=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -d "${MILABENCH_SOURCE}" ]]; then
  echo "ERROR: milabench source not found at ${MILABENCH_SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${MILABENCH_CONFIG}" ]]; then
  echo "ERROR: config not found at ${MILABENCH_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${MILABENCH_SOURCE}/benchmarks/vllm" ]]; then
  echo "ERROR: vllm bench missing at ${MILABENCH_SOURCE}/benchmarks/vllm" >&2
  exit 1
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: missing ${SBATCH_SCRIPT}" >&2
  exit 1
fi

LOG_DIR="${MILABENCH_WORKDIR}/logs"
ENV_FILE="${MILABENCH_WORKDIR}/job_env.sh"
TORCH_VENV="${MILABENCH_BASE}/venv/torch"

mkdir -p "${MILABENCH_WORKDIR}" "${MILABENCH_BASE}/runs" "${MILABENCH_BASE}/venv" "${LOG_DIR}" "${UV_CACHE_DIR}"

# num_machines for a bench: *-4nodes → 4, else 2 (matches diagnostic.yaml).
bench_nodes() {
  local name="$1"
  if [[ "${name}" == *-4nodes ]]; then
    echo 4
  else
    echo 2
  fi
}

# milabench TOML install calls resolve_vllm(required=True), which fails for arch=cpu.
# For this diagnostic smoke test we only need importable deps — versions are not important.
install_cpu_vllm_bench_deps() {
  # Any recent vLLM GitHub release page with +cpu wheels (not a package pin).
  local vllm_wheels="https://github.com/vllm-project/vllm/releases/expanded_assets/v0.19.1"
  local -a uv=(
    "${UV}" pip install --python "${TORCH_VENV}/bin/python"
    --no-build-isolation
    --index-strategy unsafe-best-match
    --index-url https://pypi.org/simple
  )

  echo "==> Installing unpinned CPU smoke deps into ${TORCH_VENV}"

  echo "    torch / voir / ray"
  "${uv[@]}" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch voir ray

  echo "    vllm (+cpu wheels via GitHub release assets)"
  "${uv[@]}" \
    --find-links "${vllm_wheels}" \
    vllm "vllm[bench]"
}

# --- sync checkouts to remote tip of current branch ---------------------------
sync_repo() {
  local repo="$1"
  local branch
  branch="$(git -C "${repo}" branch --show-current)"
  echo "==> git fetch + reset --hard origin/${branch} in ${repo}"
  git -C "${repo}" fetch --force origin "${branch}"
  git -C "${repo}" reset --hard "origin/${branch}"
}

sync_repo "${MILABENCH_SOURCE}"

# --- toolchain ----------------------------------------------------------------
module load python/"${PYTHON_VERSION}" 2>/dev/null || module load python || true

UV="${UV:-${HOME}/.local/bin/uv}"
if [[ ! -x "${UV}" ]]; then
  echo "Installing uv into ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="${HOME}/.local/bin/uv"
fi

echo "==> Preparing workspace in ${MILABENCH_WORKDIR}"
echo "    source=${MILABENCH_SOURCE}"
echo "    config=${MILABENCH_CONFIG}"
echo "    benches=${BENCHMARKS}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  # --- milabench tool venv (login has internet) -------------------------------
  if [[ ! -x "${MILABENCH_ENV}/bin/milabench" ]]; then
    echo "==> Creating milabench venv at ${MILABENCH_ENV}"
    "${UV}" venv --python="${PYTHON_VERSION}" "${MILABENCH_ENV}"
  fi
  # shellcheck disable=SC1091
  source "${MILABENCH_ENV}/bin/activate"
  if [[ "${MILABENCH_GPU_ARCH}" == "cuda" ]]; then
    "${UV}" pip install -e "${MILABENCH_SOURCE}[cuda]"
  else
    "${UV}" pip install -e "${MILABENCH_SOURCE}"
  fi

  # Placeholder system file for install/prepare on the login node.
  cat > "${MILABENCH_WORKDIR}/system.login.yaml" <<EOF
system:
  arch: ${MILABENCH_GPU_ARCH}
  nodes:
    - name: login
      ip: localhost
      hostname: $(hostname)
      user: ${USER}
      main: true
EOF

  # Seed the torch install_group venv (vLLM diagnostic uses install_group: torch).
  # milabench uses --no-build-isolation by default; voir→omegaconf→antlr4 needs
  # setuptools, and some sdists need maturin.
  echo "==> Seeding build deps in ${TORCH_VENV}"
  if [[ ! -x "${TORCH_VENV}/bin/python" ]]; then
    "${UV}" venv --python="${PYTHON_VERSION}" "${TORCH_VENV}"
  fi
  "${UV}" pip install --python "${TORCH_VENV}/bin/python" \
    setuptools wheel pip maturin poetry flit_core hatchling packaging
  "${UV}" pip install --python "${TORCH_VENV}/bin/python" \
    -e "${MILABENCH_SOURCE}/benchmate"

  SET_ARGS=("torch=${PYTORCH_VERSION}")
  if [[ "${MILABENCH_GPU_ARCH}" == "cuda" ]]; then
    SET_ARGS+=("cuda=${CUDA_VERSION}")
  fi

  # Install once for all selected benches (shared install_group / definition).
  INSTALL_SELECT="$(echo "${BENCHMARKS}" | tr ' ' ',')"
  if [[ "${MILABENCH_GPU_ARCH}" == "cpu" ]]; then
    echo "==> CPU workaround: skip milabench TOML install (broken vLLM map for arch=cpu)"
    install_cpu_vllm_bench_deps
  else
    echo "==> milabench install (--select ${INSTALL_SELECT})"
    milabench install \
      --config "${MILABENCH_CONFIG}" \
      --base "${MILABENCH_BASE}" \
      --system "${MILABENCH_WORKDIR}/system.login.yaml" \
      --set "${SET_ARGS[@]}" \
      --select "${INSTALL_SELECT}"
  fi

  if [[ ! -x "${TORCH_VENV}/bin/python" ]] || ! "${TORCH_VENV}/bin/python" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not importable in ${TORCH_VENV} after install" >&2
    "${TORCH_VENV}/bin/python" -c "import vllm" >&2 || true
    exit 1
  fi

  echo "==> milabench prepare"
  milabench prepare \
    --config "${MILABENCH_CONFIG}" \
    --base "${MILABENCH_BASE}" \
    --system "${MILABENCH_WORKDIR}/system.login.yaml" \
    --select "${INSTALL_SELECT}"
else
  echo "==> SKIP_INSTALL=1 — reusing existing workspace"
  # shellcheck disable=SC1091
  source "${MILABENCH_ENV}/bin/activate"
fi

# Shared env for sbatch jobs (paths on scratch, readable on compute nodes).
# MILABENCH_ARGS is overridden per job via sbatch --export.
cat > "${ENV_FILE}" <<EOF
# Generated by trilium_vllm_schedule.sh — do not edit by hand.
export ACCOUNT="${ACCOUNT}"
export MILABENCH_SOURCE="${MILABENCH_SOURCE}"
export MILABENCH_WORKDIR="${MILABENCH_WORKDIR}"
export MILABENCH_ENV="${MILABENCH_ENV}"
export MILABENCH_BASE="${MILABENCH_BASE}"
export MILABENCH_CONFIG="${MILABENCH_CONFIG}"
export MILABENCH_GPU_ARCH="${MILABENCH_GPU_ARCH}"
export MILABENCH_USE_TOML_DEPS=1
export PYTHONUNBUFFERED=1
export PYTHON_VERSION="${PYTHON_VERSION}"
export CUDA_VERSION="${CUDA_VERSION}"
export PYTORCH_VERSION="${PYTORCH_VERSION}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
EOF

SBATCH_EXTRA=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_EXTRA+=(--partition="${PARTITION}")
fi

declare -a JOB_IDS=()
declare -a JOB_BENCHES=()
FIRST_OUTFILE=""

for BENCH in ${BENCHMARKS}; do
  NODES_FOR_BENCH="$(bench_nodes "${BENCH}")"
  # Allow a global NODES override (applies to every job).
  if [[ -n "${NODES:-}" ]]; then
    NODES_FOR_BENCH="${NODES}"
  fi
  JOB_ARGS="--select ${BENCH}"
  JOB_NAME="vllm-${BENCH}"
  # Slurm job names are short; keep a readable prefix + bench slug.
  JOB_NAME="$(echo "${JOB_NAME}" | cut -c1-64)"

  echo "==> Submitting ${BENCH} (${NODES_FOR_BENCH} × ${CPUS_PER_TASK}-core nodes)"
  JOB_ID="$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --nodes="${NODES_FOR_BENCH}" \
    --ntasks="${NODES_FOR_BENCH}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem=0 \
    --time="${TIME}" \
    --job-name="${JOB_NAME}" \
    --output="${LOG_DIR}/${BENCH}_%j.out" \
    --error="${LOG_DIR}/${BENCH}_%j.err" \
    --export=ALL,MILABENCH_WORKDIR="${MILABENCH_WORKDIR}",MILABENCH_ARGS="${JOB_ARGS}" \
    "${SBATCH_EXTRA[@]}" \
    "${SBATCH_SCRIPT}")"

  OUTFILE="${LOG_DIR}/${BENCH}_${JOB_ID}.out"
  touch "${OUTFILE}"
  echo "    job ${JOB_ID} → ${OUTFILE}"

  JOB_IDS+=("${JOB_ID}")
  JOB_BENCHES+=("${BENCH}")
  if [[ -z "${FIRST_OUTFILE}" ]]; then
    FIRST_OUTFILE="${OUTFILE}"
    FIRST_JOB_ID="${JOB_ID}"
  fi
done

echo "Submitted ${#JOB_IDS[@]} job(s):"
for i in "${!JOB_IDS[@]}"; do
  echo "  ${JOB_BENCHES[$i]} → ${JOB_IDS[$i]}"
done

if [[ "${FOLLOW}" != "1" ]]; then
  echo "FOLLOW=0 — not waiting. Logs under ${LOG_DIR}"
  exit 0
fi

# Follow the first job; remaining jobs keep running independently.
echo "==> Following ${FIRST_OUTFILE} (other jobs still queued/running)"
(
  while [[ -n "$(squeue -j "${FIRST_JOB_ID}" -h 2>/dev/null || true)" ]]; do
    sleep 5
  done
) &
WAIT_PID=$!

tail --pid="${WAIT_PID}" -f "${FIRST_OUTFILE}"
wait "${WAIT_PID}" 2>/dev/null || true

EXIT_CODE="$(sacct -j "${FIRST_JOB_ID}" -n -P -o ExitCode 2>/dev/null | head -1 | cut -d: -f1 || true)"
echo "Remaining jobs (if any): ${JOB_IDS[*]:1}"
exit "${EXIT_CODE:-0}"
