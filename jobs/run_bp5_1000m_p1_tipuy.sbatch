#!/bin/bash
#SBATCH -J bp5_tnd_uy
#SBATCH -o bp5_tnd_uy_%j.out
#SBATCH -e bp5_tnd_uy_%j.err
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

TANDEM=/scratch2/10024/zhaochun/tandem/build_3d_1p_tipuy/app/tandem
OPTIONS=/scratch2/10024/zhaochun/tandem/tandem/examples/options

echo "num_nodes: $SLURM_JOB_NUM_NODES ntasks: $SLURM_NTASKS"
echo "TANDEM_DIAG_TIP_UY=$TANDEM_DIAG_TIP_UY"
echo "TANDEM=$TANDEM"
ls -l "$TANDEM"
echo "binary markers:"
strings "$TANDEM" | grep -E "TND-TQ-ARMED|TND-TQ-DBG|TND-TQ|TND-TIP" | head -10

ibrun $TANDEM bp5.toml --petsc \
      -options_file ${OPTIONS}/lu_mumps.cfg \
      -options_file ${OPTIONS}/rk45.cfg \
      -ts_monitor
