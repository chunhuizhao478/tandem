#include "PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include <petscpc.h>
#include <petscvec.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace tndm {

PetscLinearSolver::PetscLinearSolver(AbstractDGOperator<DomainDimension>& dgop, bool matrix_free,
                                     MGConfig const& mg_config) {
    auto const& topo = dgop.topo();
    if (matrix_free) {
        A_ = std::make_unique<PetscDGShell>(dgop);
    }

    P_ = std::make_unique<PetscDGMatrix>(dgop.block_size(), topo);
    dgop.assemble(*P_);

    // K·v diagnostic: apply assembled K to all-ones, dump for cross-code comparison.
    // All ranks participate in MatMult (collective). Only target rank writes file.
    {
        auto const* env = std::getenv("TANDEM_FIRST_STEP_DUMP");
        if (env != nullptr && std::string(env) == "1") {
            Vec ones, Kv;
            CHKERRTHROW(MatCreateVecs(P_->mat(), &ones, &Kv));
            CHKERRTHROW(VecSet(ones, 1.0));
            CHKERRTHROW(MatMult(P_->mat(), ones, Kv));

            int rank;
            MPI_Comm_rank(topo.comm(), &rank);
            int target_rank = -1;
            if (auto const* tr = std::getenv("TANDEM_FIRST_STEP_TARGET_RANK");
                tr != nullptr && *tr != '\0') {
                target_rank = std::atoi(tr);
            }

            if (target_rank < 0 || rank == target_rank) {
                std::string outdir = ".";
                if (auto const* od = std::getenv("TANDEM_FIRST_STEP_OUTPUT_DIR");
                    od != nullptr && *od != '\0') {
                    outdir = od;
                }
                std::ostringstream path;
                path << outdir;
                if (!outdir.empty() && outdir.back() != '/') path << "/";
                path << "first_step_Kv_r" << rank << ".csv";

                PetscInt local_size;
                CHKERRTHROW(VecGetLocalSize(Kv, &local_size));
                const PetscScalar *kv_arr;
                CHKERRTHROW(VecGetArrayRead(Kv, &kv_arr));

                std::ofstream out(path.str(), std::ios::trunc);
                out << "local_dof,Kv_value\n";
                out << std::setprecision(17);
                for (PetscInt i = 0; i < local_size; i++) {
                    out << i << "," << kv_arr[i] << "\n";
                }
                CHKERRTHROW(VecRestoreArrayRead(Kv, &kv_arr));
                std::cout << "[K-DIAG] K·1 dumped to " << path.str()
                          << " (" << local_size << " local DOFs)\n";
            }

            // Print global norms (rank 0 only, but computed collectively)
            PetscReal norm1, norm2, norminf;
            CHKERRTHROW(VecNorm(Kv, NORM_1, &norm1));
            CHKERRTHROW(VecNorm(Kv, NORM_2, &norm2));
            CHKERRTHROW(VecNorm(Kv, NORM_INFINITY, &norminf));
            if (rank == 0) {
                std::cout << "[K-DIAG] ||K·1||_1   = " << std::setprecision(15) << norm1 << "\n";
                std::cout << "[K-DIAG] ||K·1||_2   = " << std::setprecision(15) << norm2 << "\n";
                std::cout << "[K-DIAG] ||K·1||_inf = " << std::setprecision(15) << norminf << "\n";
            }

            CHKERRTHROW(VecDestroy(&ones));
            CHKERRTHROW(VecDestroy(&Kv));
        }
    }

    b_ = std::make_unique<PetscVector>(dgop.block_size(), topo.numLocalElements(), topo.comm());
    x_ = std::make_unique<PetscVector>(*b_);
    dgop.rhs(*b_);

    CHKERRTHROW(KSPCreate(topo.comm(), &ksp_));
    CHKERRTHROW(KSPSetType(ksp_, KSPCG));
    if (matrix_free) {
        CHKERRTHROW(KSPSetOperators(ksp_, A_->mat(), P_->mat()));
    } else {
        CHKERRTHROW(KSPSetOperators(ksp_, P_->mat(), P_->mat()));
    }
    CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

    PC pc;
    CHKERRTHROW(KSPGetPC(ksp_, &pc));
    CHKERRTHROW(PCSetFromOptions(pc));
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(PCMG):
        setup_mg(dgop, pc, mg_config);
        break;
    default:
        break;
    };

    CHKERRTHROW(KSPSetFromOptions(ksp_));
}

PetscLinearSolver::~PetscLinearSolver() {
    for (auto&& A : mat_cleanup) {
        MatDestroy(&A);
    }
    KSPDestroy(&ksp_);
}

void PetscLinearSolver::setup_mg(AbstractDGOperator<DomainDimension>& dgop, PC pc,
                                 MGConfig const& mg_config) {
    auto i_op = dgop.interpolation_operator();

    auto level_degree = mg_config.levels(i_op->max_degree());
    unsigned nlevels = level_degree.size();

    int rank;
    MPI_Comm_rank(dgop.topo().comm(), &rank);
    if (rank == 0) {
        std::cout << "Multigrid P-levels: ";
        for (auto level : level_degree) {
            std::cout << level << " ";
        }
        std::cout << std::endl;
    }

    CHKERRTHROW(PCMGSetLevels(pc, nlevels, nullptr));
    CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
    KSP smooth;
    CHKERRTHROW(PCMGGetSmoother(pc, nlevels - 1, &smooth));
    if (A_) {
        CHKERRTHROW(KSPSetOperators(smooth, A_->mat(), P_->mat()));
    } else {
        CHKERRTHROW(KSPSetOperators(smooth, P_->mat(), P_->mat()));
    }
    Mat A_lp1 = P_->mat();
    for (int l = nlevels - 2; l >= 0; --l) {
        unsigned to_degree = level_degree[l + 1];
        unsigned from_degree = level_degree[l];
        auto I = PetscInterplMatrix(i_op->block_size(to_degree), i_op->block_size(from_degree),
                                    dgop.topo());
        i_op->assemble(to_degree, from_degree, I);
        CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, I.mat()));

        // Construct coarse level operator via A_c = Pt A P
        // Mat A_l;
        // CHKERRTHROW(MatPtAP(A_lp1, I.mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_l));

        // PETSc manpage for MatPtAP() states that For matrix types without a
        // special implementation of PtAP, MatPtAP() fallbacks back to
        // using MatMatMult() followed by MatTransposeMatMult().
        // The following fall back will always work
        Mat A_l, AP, Pt;
        CHKERRTHROW(MatMatMult(A_lp1, I.mat(), MAT_INITIAL_MATRIX, 1, &AP));
        CHKERRTHROW(MatTranspose(I.mat(), MAT_INITIAL_MATRIX, &Pt));
        CHKERRTHROW(MatMatMult(Pt, AP, MAT_INITIAL_MATRIX, 1, &A_l));

        // When using MatType = aijhipsparse thip loses the blocksize somewhere during
        // the computation of AP = A P and Ac = Pt AP - hence we force it here.
        // The convergence of GAMG is strongly connected with preserving the block size.
        CHKERRTHROW(MatSetBlockSize(A_l, (PetscInt)i_op->block_size(from_degree)));

        CHKERRTHROW(MatDestroy(&Pt));
        CHKERRTHROW(MatDestroy(&AP));

        CHKERRTHROW(PCMGGetSmoother(pc, l, &smooth));
        CHKERRTHROW(KSPSetOperators(smooth, A_l, A_l));
        mat_cleanup.push_back(A_l);
        A_lp1 = A_l;
    }
}

void PetscLinearSolver::warmup() { warmup_ksp(ksp_); }

void PetscLinearSolver::warmup_ksp(KSP ksp) {
    PC pc;
    CHKERRTHROW(KSPSetUp(ksp));
    CHKERRTHROW(KSPSetUpOnBlocks(ksp));
    CHKERRTHROW(KSPGetPC(ksp, &pc));
    warmup_sub_pcs(pc);
}

void PetscLinearSolver::warmup_sub_pcs(PC pc) {
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(PCCOMPOSITE):
        warmup_composite(pc);
        break;
    case HASH_DEF(PCMG):
        warmup_mg(pc);
        break;
    default:
        break;
    };
}

void PetscLinearSolver::warmup_composite(PC pc) {
    PetscInt nc;
    CHKERRTHROW(PCCompositeGetNumberPC(pc, &nc));
    for (PetscInt n = 0; n < nc; ++n) {
        PC sub;
        CHKERRTHROW(PCCompositeGetPC(pc, n, &sub));
        CHKERRTHROW(PCSetUp(sub));
        CHKERRTHROW(PCSetUpOnBlocks(sub));
        warmup_sub_pcs(sub);
    }
}

void PetscLinearSolver::warmup_mg(PC pc) {
    PetscInt levels;
    CHKERRTHROW(PCMGGetLevels(pc, &levels));
    for (PetscInt level = 0; level < levels; ++level) {
        KSP smoother;
        CHKERRTHROW(PCMGGetSmoother(pc, level, &smoother));
        warmup_ksp(smoother);
    }
}

void PetscLinearSolver::dump() const {
    PetscViewer viewer;
    CHKERRTHROW(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    CHKERRTHROW(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
    CHKERRTHROW(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "A.bin"));
    CHKERRTHROW(MatView(P_->mat(), viewer));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "b.bin"));
    CHKERRTHROW(VecView(b_->vec(), viewer));

    CHKERRTHROW(PetscViewerDestroy(&viewer));
}

} // namespace tndm
