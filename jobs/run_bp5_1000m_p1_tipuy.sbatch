#!/bin/bash
set -euo pipefail

# Self-contained Tandem BP5 tip-traction diagnostic run.
# This script is intentionally opinionated so the binary, working directory,
# and output files cannot silently drift apart.

# Fixed scratch layout
ROOT=/scratch2/10024/zhaochun/tandem
RUN_DIR=${ROOT}/bp5_1000m_p1
BUILD_DIR=${ROOT}/build_3d_1p_tipdiag
TANDEM=${BUILD_DIR}/app/tandem
OPTIONS=${ROOT}/tandem/examples/options

#SBATCH -J bp5_tnd_uy
#SBATCH -o /scratch2/10024/zhaochun/tandem/bp5_1000m_p1/bp5_tnd_uy_%j.out
#SBATCH -e /scratch2/10024/zhaochun/tandem/bp5_1000m_p1/bp5_tnd_uy_%j.err
#SBATCH -p development
#SBATCH -N 8
#SBATCH -n 400
#SBATCH -t 02:00:00
#SBATCH -A EAR20006
#SBATCH --chdir=/scratch2/10024/zhaochun/tandem/bp5_1000m_p1

# Tandem BP5 p=1 1000m: [TND-TIP] + [TND-TQ] diagnostics
# Compare normal traction at tip faces with MFEM [TIP-UY].

module load gcc/9.1.0
module load impi/19.0.9

export OMP_NUM_THREADS=1
export TANDEM_DIAG_TIP_UY=1

cd "$RUN_DIR"

echo "pwd=$(pwd)"
echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "num_nodes: $SLURM_JOB_NUM_NODES ntasks: $SLURM_NTASKS"
echo "TANDEM_DIAG_TIP_UY=$TANDEM_DIAG_TIP_UY"
echo "RUN_DIR=$RUN_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "TANDEM=$TANDEM"
ls -l "$TANDEM"
echo "binary markers:"
strings "$TANDEM" | grep -E "TND-TQ-ARMED|TND-TQ-DBG|TND-TQ|TND-TIP" | head -10
echo "input files:"
ls -l bp5.toml bp5.lua bp5.msh

# Fail fast if the binary is stale. This is the main source of confusion in
# previous runs, so make the job stop immediately instead of producing junk.
if ! strings "$TANDEM" | grep -q "TND-TQ-ARMED"; then
  echo "ERROR: diagnostic binary marker TND-TQ-ARMED not found in $TANDEM" >&2
  exit 2
fi
if ! strings "$TANDEM" | grep -q "cx="; then
  echo "ERROR: left-tip coordinate markers not found in $TANDEM" >&2
  exit 2
fi

ibrun "$TANDEM" bp5.toml --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor
