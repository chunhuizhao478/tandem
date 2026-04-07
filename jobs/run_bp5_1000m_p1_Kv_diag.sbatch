#!/bin/bash
#SBATCH -J bp5_tnd_Kv
#SBATCH -o /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step/bp5_tnd_Kv_%j.out
#SBATCH -e /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step/bp5_tnd_Kv_%j.err
#SBATCH -p development
#SBATCH -N 8
#SBATCH -n 400
#SBATCH -t 02:00:00
#SBATCH -A EAR20006
#SBATCH --chdir=/scratch2/10024/zhaochun/tandem/bp5_1000m_p1_first_step

set -euo pipefail

# Tandem BP5: K·1 norm diagnostic for cross-code comparison with MFEM.
# Rebuilds from feature/tip-diagnostic branch, then runs first-step dump.

ROOT=/scratch2/10024/zhaochun/tandem
RUN_DIR=${ROOT}/bp5_1000m_p1_first_step
BUILD_DIR=${ROOT}/build_3d_1p_normaldiag
TANDEM=${BUILD_DIR}/app/tandem
OPTIONS=${ROOT}/tandem/examples/options
CONFIG=bp5_normal_diag.toml

module load gcc/9.1.0
module load impi/19.0.9

# Source already checked out on login node.
# Build must be done BEFORE submitting (cmake needs newer gcc).
# On login node:
#   module load gcc/13.2.0   # or whatever was used to build originally
#   cd ${BUILD_DIR} && cmake --build . -j8
#
# This sbatch assumes the binary is already rebuilt.
cd ${ROOT}/tandem
echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"

[ -x "$TANDEM" ] || { echo "ERROR: $TANDEM not found. Rebuild first."; exit 1; }

export OMP_NUM_THREADS=1
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

# Short run: just a few steps to capture K·1 norms and first-step dump
ibrun "$TANDEM" "$CONFIG" --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor \
      -ts_max_steps 10

echo "First-step dump files:"
ls -lh "${RUN_DIR}"/first_step_*.csv 2>/dev/null || echo "  (none found)"
