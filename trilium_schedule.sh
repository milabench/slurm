#!/bin/bash
# Prepare milabench + torchsrun on a Trillium CPU login node, then submit the
# compute job. Compute nodes have no internet and cannot write $HOME/$PROJECT
# (see https://docs.alliancecan.ca/wiki/Trillium).
#
# Usage (on trillium.alliancecan.ca — CPU login, not GPU):
#   ./trilium_schedule.sh
#
# Node count: queries idle CPU nodes via sinfo and requests half (min 2).
# Override with NODES=N if you want a fixed size.
#
# Optional env overrides:
#   MILABENCH_SOURCE / MILABENCH_WORKDIR / ACCOUNT / NODES / MAX_NODES / TIME
#   PYTHON_VERSION / CUDA_VERSION / PYTORCH_VERSION

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/trilium_run.sbatch"

# --- defaults -----------------------------------------------------------------
export ACCOUNT="${ACCOUNT:-rrg-bengioy-ad}"
export TIME="${TIME:-1:00:00}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
export CUDA_VERSION="${CUDA_VERSION:-130}"
export PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
# CUDA wheels still fine for a CPU/gloo torchsrun smoke test.
export MILABENCH_GPU_ARCH="${MILABENCH_GPU_ARCH:-cuda}"
export MILABENCH_SELECT="${MILABENCH_SELECT:-torchsrun}"
# Trillium CPU nodes are whole 192-core nodes.
export CPUS_PER_TASK="${CPUS_PER_TASK:-192}"
export PARTITION="${PARTITION:-}"
export MAX_NODES="${MAX_NODES:-}"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: \$SCRATCH is unset. Run this on a Trillium login node." >&2
  exit 1
fi

# Scripts live in <repo>/slurm/; milabench checkout is <repo>/milabench.
# Scripts live in <repo>/slurm/; milabench checkout is <repo>/milabench.
export MILABENCH_SOURCE="${MILABENCH_SOURCE:-${SCRIPT_DIR}/../milabench}"
MILABENCH_SOURCE="$(cd "${MILABENCH_SOURCE}" && pwd)"
export MILABENCH_SOURCE
export MILABENCH_WORKDIR="${MILABENCH_WORKDIR:-${SCRATCH}/milabench_torchsrun}"
export MILABENCH_ENV="${MILABENCH_ENV:-${MILABENCH_WORKDIR}/.env/${PYTHON_VERSION}}"
export MILABENCH_BASE="${MILABENCH_BASE:-${MILABENCH_WORKDIR}/results}"
export MILABENCH_CONFIG="${MILABENCH_CONFIG:-${MILABENCH_SOURCE}/config/diagnostic/diagnostic.yaml}"
export MILABENCH_ARGS="${MILABENCH_ARGS:---select ${MILABENCH_SELECT}}"
export PYTHONUNBUFFERED=1
export MILABENCH_USE_TOML_DEPS=1

if [[ ! -d "${MILABENCH_SOURCE}" ]]; then
  echo "ERROR: milabench source not found at ${MILABENCH_SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${MILABENCH_CONFIG}" ]]; then
  echo "ERROR: config not found at ${MILABENCH_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${MILABENCH_SOURCE}/benchmarks/torchsrun" ]]; then
  echo "ERROR: torchsrun bench missing at ${MILABENCH_SOURCE}/benchmarks/torchsrun" >&2
  exit 1
fi

LOG_DIR="${MILABENCH_WORKDIR}/logs"
ENV_FILE="${MILABENCH_WORKDIR}/job_env.sh"

mkdir -p "${MILABENCH_WORKDIR}" "${MILABENCH_BASE}/runs" "${LOG_DIR}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: missing ${SBATCH_SCRIPT}" >&2
  exit 1
fi

# --- discover idle CPU nodes → request half -----------------------------------
# Run on the CPU login; sinfo here only sees the CPU subcluster.
count_idle_cpu_nodes() {
  local sinfo_args=(-N -h -t idle -o '%N')
  if [[ -n "${PARTITION}" ]]; then
    sinfo_args+=(-p "${PARTITION}")
  fi
  # Prefer idle; fall back to "idle*" / available counts if needed.
  local idle
  idle="$(sinfo "${sinfo_args[@]}" 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -z "${idle}" || "${idle}" -eq 0 ]]; then
    # %A = available/other node counts for the partition
    local avail_fmt part_args=()
    [[ -n "${PARTITION}" ]] && part_args=(-p "${PARTITION}")
    avail_fmt="$(sinfo -h "${part_args[@]}" -o '%A' 2>/dev/null | head -1 | cut -d/ -f1 || true)"
    idle="${avail_fmt:-0}"
  fi
  echo "${idle}"
}

if [[ -z "${NODES:-}" ]]; then
  IDLE_NODES="$(count_idle_cpu_nodes)"
  NODES=$((IDLE_NODES / 2))
  if [[ "${NODES}" -lt 2 ]]; then
    NODES=2
  fi
  if [[ -n "${MAX_NODES}" && "${NODES}" -gt "${MAX_NODES}" ]]; then
    NODES="${MAX_NODES}"
  fi
  echo "==> Idle CPU nodes: ${IDLE_NODES} → requesting half: ${NODES}"
else
  echo "==> Using explicit NODES=${NODES}"
fi
export NODES

# Warn if we look like the GPU login.
if hostname 2>/dev/null | grep -qiE 'trig|gpu'; then
  echo "WARNING: CPU jobs should be submitted from trillium.alliancecan.ca" >&2
fi

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
echo "    select=${MILABENCH_SELECT}"

# --- milabench tool venv (login has internet) ---------------------------------
if [[ ! -x "${MILABENCH_ENV}/bin/milabench" ]]; then
  echo "==> Creating milabench venv at ${MILABENCH_ENV}"
  "${UV}" venv --python="${PYTHON_VERSION}" "${MILABENCH_ENV}"
fi
# shellcheck disable=SC1091
source "${MILABENCH_ENV}/bin/activate"
"${UV}" pip install -e "${MILABENCH_SOURCE}[${MILABENCH_GPU_ARCH}]"

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

echo "==> milabench install (torchsrun / torch group)"
milabench install \
  --config "${MILABENCH_CONFIG}" \
  --base "${MILABENCH_BASE}" \
  --system "${MILABENCH_WORKDIR}/system.login.yaml" \
  --set "cuda=${CUDA_VERSION}" "torch=${PYTORCH_VERSION}" \
  ${MILABENCH_ARGS}

echo "==> milabench prepare"
milabench prepare \
  --config "${MILABENCH_CONFIG}" \
  --base "${MILABENCH_BASE}" \
  --system "${MILABENCH_WORKDIR}/system.login.yaml" \
  ${MILABENCH_ARGS}

# Shared env for the sbatch job (paths on scratch, readable on compute nodes).
cat > "${ENV_FILE}" <<EOF
# Generated by trilium_schedule.sh — do not edit by hand.
export ACCOUNT="${ACCOUNT}"
export NODES="${NODES}"
export MILABENCH_SOURCE="${MILABENCH_SOURCE}"
export MILABENCH_WORKDIR="${MILABENCH_WORKDIR}"
export MILABENCH_ENV="${MILABENCH_ENV}"
export MILABENCH_BASE="${MILABENCH_BASE}"
export MILABENCH_CONFIG="${MILABENCH_CONFIG}"
export MILABENCH_ARGS="${MILABENCH_ARGS}"
export MILABENCH_GPU_ARCH="${MILABENCH_GPU_ARCH}"
export MILABENCH_USE_TOML_DEPS=1
export PYTHONUNBUFFERED=1
export PYTHON_VERSION="${PYTHON_VERSION}"
export CUDA_VERSION="${CUDA_VERSION}"
export PYTORCH_VERSION="${PYTORCH_VERSION}"
EOF

SBATCH_EXTRA=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_EXTRA+=(--partition="${PARTITION}")
fi

echo "==> Submitting CPU torchsrun job (${NODES} × ${CPUS_PER_TASK}-core nodes)"
JOB_ID="$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --nodes="${NODES}" \
  --ntasks="${NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem=0 \
  --time="${TIME}" \
  --job-name="torchsrun-cpu" \
  --output="${LOG_DIR}/torchsrun_%j.out" \
  --error="${LOG_DIR}/torchsrun_%j.err" \
  --export=ALL,MILABENCH_WORKDIR="${MILABENCH_WORKDIR}" \
  "${SBATCH_EXTRA[@]}" \
  "${SBATCH_SCRIPT}")"

echo "Submitted job ${JOB_ID}"
echo "Logs: ${LOG_DIR}/torchsrun_${JOB_ID}.out"

OUTFILE="${LOG_DIR}/torchsrun_${JOB_ID}.out"
ERRFILE="${LOG_DIR}/torchsrun_${JOB_ID}.err"

if command -v jq >/dev/null 2>&1; then
  SLURM_OUT="$(scontrol show job "${JOB_ID}" --json 2>/dev/null \
    | jq -r '.jobs[0].standard_output // empty' || true)"
  if [[ -n "${SLURM_OUT}" && "${SLURM_OUT}" != "null" ]]; then
    OUTFILE="${SLURM_OUT}"
  fi
fi

echo "==> Waiting for job ${JOB_ID} (tail -f when output appears)"
while true; do
  STATE="$(squeue -j "${JOB_ID}" -h -o '%T' 2>/dev/null || true)"
  if [[ -z "${STATE}" ]]; then
    echo "Job ${JOB_ID} left the queue."
    [[ -f "${OUTFILE}" ]] && cat "${OUTFILE}"
    [[ -f "${ERRFILE}" && -s "${ERRFILE}" ]] && { echo "--- stderr ---"; cat "${ERRFILE}"; }
    EXIT_CODE="$(sacct -j "${JOB_ID}" -n -P -o ExitCode 2>/dev/null | head -1 | cut -d: -f1 || true)"
    exit "${EXIT_CODE:-0}"
  fi

  case "${STATE}" in
    RUNNING|COMPLETING)
      if [[ -f "${OUTFILE}" ]]; then
        echo "Job ${JOB_ID} is ${STATE}; following ${OUTFILE}"
        tail -n +1 -F "${OUTFILE}" &
        TAIL_PID=$!
        while [[ -n "$(squeue -j "${JOB_ID}" -h 2>/dev/null || true)" ]]; do
          sleep 5
        done
        sleep 1
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
        [[ -f "${ERRFILE}" && -s "${ERRFILE}" ]] && { echo "--- stderr ---"; cat "${ERRFILE}"; }
        EXIT_CODE="$(sacct -j "${JOB_ID}" -n -P -o ExitCode 2>/dev/null | head -1 | cut -d: -f1 || true)"
        EXIT_CODE="${EXIT_CODE:-0}"
        echo "Job ${JOB_ID} finished (ExitCode=${EXIT_CODE})"
        exit "${EXIT_CODE}"
      fi
      ;;
    PENDING|CONFIGURING|REQUEUED)
      echo "  [${STATE}] waiting for allocation..."
      ;;
    *)
      echo "  [${STATE}] ..."
      ;;
  esac
  sleep 10
done
