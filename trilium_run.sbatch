#!/bin/bash
# Run milabench torchsrun on Trillium CPU compute nodes (no GPUs).
# Prep (venv/data/install) is done on the login node by trilium_schedule.sh —
# compute nodes have no internet and cannot write $HOME/$PROJECT.
# Docs: https://docs.alliancecan.ca/wiki/Trillium
#
# Prefer submitting via ./trilium_schedule.sh (it picks half the idle CPU nodes).
# Direct submit example (after schedule prep):
#   sbatch --account=rrg-bengioy-ad --nodes=N --ntasks=N --ntasks-per-node=1 \
#     --cpus-per-task=192 --mem=0 --time=1:00:00 \
#     --export=ALL,MILABENCH_WORKDIR=$SCRATCH/milabench_torchsrun \
#     trilium_run.sbatch

#SBATCH --job-name=torchsrun-cpu
#SBATCH --account=rrg-bengioy-ad
# Resource sizes are overridden by trilium_schedule.sh; placeholders only.
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=1:00:00

set -euo pipefail

echo "=== Trillium CPU torchsrun job ${SLURM_JOB_ID:-unknown} ==="
echo "Nodes (${SLURM_NNODES:-?}):- ${SLURM_JOB_NODELIST:-}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
hostname
date

if [[ -z "${MILABENCH_WORKDIR:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    MILABENCH_WORKDIR="${SCRATCH}/milabench_torchsrun"
  else
    echo "ERROR: set MILABENCH_WORKDIR (or SCRATCH) to the prepared workspace" >&2
    exit 1
  fi
fi

ENV_FILE="${MILABENCH_WORKDIR}/job_env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "WARNING: ${ENV_FILE} missing; using environment / defaults" >&2
fi

: "${MILABENCH_SOURCE:?MILABENCH_SOURCE unset — run trilium_schedule.sh first}"
: "${MILABENCH_ENV:?MILABENCH_ENV unset — run trilium_schedule.sh first}"
: "${MILABENCH_BASE:?MILABENCH_BASE unset — run trilium_schedule.sh first}"
: "${MILABENCH_CONFIG:?MILABENCH_CONFIG unset — run trilium_schedule.sh first}"

export MILABENCH_ARGS="${MILABENCH_ARGS:---select torchsrun}"
export MILABENCH_USE_TOML_DEPS="${MILABENCH_USE_TOML_DEPS:-1}"
export MILABENCH_GPU_ARCH="${MILABENCH_GPU_ARCH:-cpu}"
export PYTHONUNBUFFERED=1
# Keep writes on scratch / job RAM disk — never $HOME on Trillium compute.
export TMPDIR="${SLURM_TMPDIR:-${MILABENCH_WORKDIR}/tmp}"
export XDG_CACHE_HOME="${MILABENCH_BASE}/cache"
mkdir -p "${TMPDIR}" "${MILABENCH_BASE}/runs" "${XDG_CACHE_HOME}"

module load python/"${PYTHON_VERSION:-3.12}" 2>/dev/null || module load python || true

if [[ ! -f "${MILABENCH_ENV}/bin/activate" ]]; then
  echo "ERROR: milabench venv missing at ${MILABENCH_ENV}" >&2
  echo "Run trilium_schedule.sh on the CPU login node first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${MILABENCH_ENV}/bin/activate"

if [[ ! -x "$(command -v milabench)" ]]; then
  echo "ERROR: milabench not on PATH after activating ${MILABENCH_ENV}" >&2
  exit 1
fi

SYSTEM_YAML="${MILABENCH_BASE}/runs/system.${SLURM_JOB_ID}.yaml"
echo "==> Generating system.yaml from Slurm allocation"
milabench slurm system --output "${SYSTEM_YAML}"
cat "${SYSTEM_YAML}"

echo "==> milabench run ${MILABENCH_ARGS}"
# No install/prepare here: compute nodes cannot reach the internet.
# torchsrun uses all nodes listed in system.yaml (gloo on CPU).
milabench run \
  --config "${MILABENCH_CONFIG}" \
  --base "${MILABENCH_BASE}" \
  --system "${SYSTEM_YAML}" \
  ${MILABENCH_ARGS}

echo "==> Done"
date
ls -la "${MILABENCH_BASE}/runs" || true
