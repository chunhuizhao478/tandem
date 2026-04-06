#!/bin/bash
#SBATCH -J bp5_tnd_norm
#SBATCH -o /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag/bp5_tnd_norm_%j.out
#SBATCH -e /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag/bp5_tnd_norm_%j.err
#SBATCH -p normal
#SBATCH -N 8
#SBATCH -n 400
#SBATCH -t 48:00:00
#SBATCH -A EAR20006
#SBATCH --chdir=/scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag

set -euo pipefail

# Self-contained Tandem BP5 normal-traction diagnostic run.
# Adds fault probes at depth=40km (fault tip) to compare sigma_n_eff
# with MFEM's traced blowup DOFs.

# Fixed scratch layout
ROOT=/scratch2/10024/zhaochun/tandem
RUN_DIR=${ROOT}/bp5_1000m_p1_normal_diag
BUILD_DIR=${ROOT}/build_3d_1p_normaldiag
TANDEM=${BUILD_DIR}/app/tandem
OPTIONS=${ROOT}/tandem/examples/options
SRC_DIR=${ROOT}/tandem/examples/tandem/3d
CONFIG=bp5_normal_diag.toml

module load gcc/9.1.0
module load impi/19.0.9

export OMP_NUM_THREADS=1
export TANDEM_DIAG_TIP_UY=1

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Copy input files so the run directory is self-contained (no symlinks)
cp -f "${SRC_DIR}/bp5.lua"              "$RUN_DIR/bp5.lua"
cp -f "${SRC_DIR}/bp5.msh"             "$RUN_DIR/bp5.msh"
cp -f "${SRC_DIR}/${CONFIG}"            "$RUN_DIR/${CONFIG}"

echo "pwd=$(pwd)"
echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "num_nodes: $SLURM_JOB_NUM_NODES ntasks: $SLURM_NTASKS"
echo "TANDEM_DIAG_TIP_UY=$TANDEM_DIAG_TIP_UY"
echo "RUN_DIR=$RUN_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "TANDEM=$TANDEM"
echo "CONFIG=$CONFIG"
ls -l "$TANDEM"
echo "binary markers:"
strings "$TANDEM" | grep -E "TND-TQ-ARMED|TND-TQ-DBG|TND-TQ|TND-TIP" | head -10 || true
echo "input files:"
ls -l "$CONFIG" bp5.lua bp5.msh

# Fail fast if the binary is stale
if ! grep -a -q "TND-TQ-ARMED" "$TANDEM"; then
  echo "ERROR: diagnostic binary marker TND-TQ-ARMED not found in $TANDEM" >&2
  exit 2
fi

ibrun "$TANDEM" "$CONFIG" --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor
