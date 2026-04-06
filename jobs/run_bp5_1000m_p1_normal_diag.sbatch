#!/bin/bash
#SBATCH -J bp5_tnd_norm
#SBATCH -o /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag/bp5_tnd_norm_%j.out
#SBATCH -e /scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag/bp5_tnd_norm_%j.err
#SBATCH -p development
#SBATCH -N 8
#SBATCH -n 400
#SBATCH -t 02:00:00
#SBATCH -A EAR20006
#SBATCH --chdir=/scratch2/10024/zhaochun/tandem/bp5_1000m_p1_normal_diag

set -euo pipefail

ROOT=/scratch2/10024/zhaochun/tandem
SRC_DIR=${ROOT}/tandem/examples/tandem/3d
RUN_DIR=${ROOT}/bp5_1000m_p1_normal_diag
BUILD_DIR=${ROOT}/build_3d_1p_tipdiag
TANDEM=${BUILD_DIR}/app/tandem
OPTIONS=${ROOT}/tandem/examples/options
CONFIG=bp5_normal_diag.toml

module load gcc/9.1.0
module load impi/19.0.9

export OMP_NUM_THREADS=1
export TANDEM_DIAG_TIP_UY=1

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

ln -sfn "${SRC_DIR}/${CONFIG}" "${CONFIG}"
ln -sfn "${SRC_DIR}/bp5.lua" bp5.lua
ln -sfn "${SRC_DIR}/bp5.msh" bp5.msh

echo "pwd=$(pwd)"
echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "num_nodes: $SLURM_JOB_NUM_NODES ntasks: $SLURM_NTASKS"
echo "TANDEM_DIAG_TIP_UY=$TANDEM_DIAG_TIP_UY"
echo "SRC_DIR=$SRC_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "TANDEM=$TANDEM"
echo "CONFIG=$CONFIG"
ls -l "$TANDEM"
echo "binary markers:"
strings "$TANDEM" | grep -E "TND-TQ-ARMED|TND-TQ-DBG|TND-TQ|TND-TIP" | head -10 || true

echo "input files:"
ls -l "$CONFIG" bp5.lua bp5.msh

if ! strings "$TANDEM" | grep -q "TND-TQ-ARMED"; then
  echo "ERROR: diagnostic binary marker TND-TQ-ARMED not found in $TANDEM" >&2
  exit 2
fi

ibrun "$TANDEM" "$CONFIG" --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor
