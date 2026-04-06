#!/bin/bash
#SBATCH -J bp5_tnd_fs
#SBATCH -o /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step/bp5_tnd_fs_%j.out
#SBATCH -e /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step/bp5_tnd_fs_%j.err
#SBATCH -p development
#SBATCH -N 8
#SBATCH -n 400
#SBATCH -t 02:00:00
#SBATCH -A EAR20006
#SBATCH --chdir=/scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step

set -euo pipefail

# Tandem BP5: First-step dump for cross-code comparison with MFEM.
# Dumps RHS, displacement jumps, and traction at fault-tip faces
# in the coordinate window x=[-37.5,-35.5], depth=[38.5,40.5] km.
# Compare output CSVs with MFEM's first_step_*.csv files.

ROOT=/scratch2/10024/zhaochun/tandem
RUN_DIR=${ROOT}/bp5_1000m_p1_first_step
BUILD_DIR=${ROOT}/build_3d_1p_normaldiag
TANDEM=${BUILD_DIR}/app/tandem
OPTIONS=${ROOT}/tandem/examples/options
CONFIG=bp5_normal_diag.toml

module load gcc/9.1.0
module load impi/19.0.9

export OMP_NUM_THREADS=1
export TANDEM_DIAG_TIP_UY=1
export TANDEM_FIRST_STEP_DUMP=1
export TANDEM_FIRST_STEP_TARGET_RANK=40
export TANDEM_FIRST_STEP_OUTPUT_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Input files must already exist in RUN_DIR
for f in bp5.lua bp5.msh "${CONFIG}"; do
    [ -f "$f" ] || { echo "ERROR: missing $f in $RUN_DIR"; exit 1; }
done

echo "pwd=$(pwd)"
echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "num_nodes: $SLURM_JOB_NUM_NODES ntasks: $SLURM_NTASKS"
echo "TANDEM_FIRST_STEP_DUMP=$TANDEM_FIRST_STEP_DUMP"
echo "TANDEM_FIRST_STEP_OUTPUT_DIR=$TANDEM_FIRST_STEP_OUTPUT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "TANDEM=$TANDEM"
echo "CONFIG=$CONFIG"
ls -l "$TANDEM"

# Fail fast if binary is stale
if ! grep -a -q "TND-FS-ARMED" "$TANDEM"; then
    echo "ERROR: TND-FS-ARMED marker not found in $TANDEM" >&2
    exit 2
fi

# Short run: just a few steps to capture the first-step dump
ibrun "$TANDEM" "$CONFIG" --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor \
      -ts_max_steps 10

echo "First-step dump files:"
ls -lh "${RUN_DIR}"/first_step_*.csv 2>/dev/null || echo "  (none found)"
