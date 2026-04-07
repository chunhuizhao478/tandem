#include "SeasQDOperator.h"

#include "form/RefElement.h"
#include "localoperator/Elasticity.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <petscvec.h>

namespace tndm {

SeasQDOperator::SeasQDOperator(std::unique_ptr<dg_t> dgop,
                               std::unique_ptr<AbstractAdapterOperator> adapter,
                               std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free,
                               MGConfig const& mg_config)
    : dgop_(std::move(dgop)), linear_solver_(*dgop_, matrix_free, mg_config),
      adapter_(std::move(adapter)), friction_(std::move(friction)),
      disp_scatter_(dgop_->topo().elementScatterPlan()),
      disp_ghost_(disp_scatter_.recv_prototype<double>(dgop_->block_size(), ALIGNMENT)),
      state_scatter_(adapter_->fault_map().scatter_plan()),
      state_ghost_(state_scatter_.recv_prototype<double>(friction_->block_size(), ALIGNMENT)),
      traction_(adapter_->traction_block_size(), adapter_->num_local_elements(), adapter_->comm()) {
}

void SeasQDOperator::set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    fun_boundary_ = std::move(fun);
}

void SeasQDOperator::initial_condition(BlockVector& state) {
    friction_->pre_init(state);

    update_ghost_state(state);
    solve(0.0, make_state_view(state));
    update_traction(make_state_view(state));

    friction_->init(0.0, traction_, state);
}

void SeasQDOperator::rhs(double time, BlockVector const& state, BlockVector& result) {
    Elasticity::SetDiagTime(time);
    update_ghost_state(state);
    solve(time, make_state_view(state));
    update_traction(make_state_view(state));

    friction_->rhs(time, traction_, state, result);
}

void SeasQDOperator::update_internal_state(double time, BlockVector const& state,
                                           bool state_changed_since_last_rhs, bool require_traction,
                                           bool require_displacement) {
    bool require_solve = state_changed_since_last_rhs && (require_traction || require_displacement);
    if (!require_solve) {
        return;
    }

    update_ghost_state(state);
    solve(time, make_state_view(state));
    if (require_traction) {
        update_traction(make_state_view(state));
    }
}

void SeasQDOperator::solve(double time, BlockView const& state_view) {
    dgop_->set_slip(adapter_->slip_bc(state_view));
    if (fun_boundary_) {
        dgop_->set_dirichlet((*fun_boundary_)(time));
    }
    linear_solver_.update_rhs(*dgop_);
    linear_solver_.solve();

    // Print global norms of b and u at every RK45 stage of first step.
    // All ranks participate in VecNorm (collective). Max 10 dumps.
    {
        static int norm_count = 0;
        auto const* env = std::getenv("TANDEM_FIRST_STEP_DUMP");
        if (norm_count < 10 && env != nullptr && std::string(env) == "1" && time > 0.0) {
            norm_count++;
            int rank;
            MPI_Comm_rank(comm(), &rank);
            if (rank == 0) {
                std::cout << "[NORM] Stage " << norm_count
                          << " at time = " << std::setprecision(17) << time << "\n";
            }

            auto print_vec_norm = [&](const char *label, Vec v) {
                PetscReal n1, n2, ninf;
                VecNorm(v, NORM_1, &n1);
                VecNorm(v, NORM_2, &n2);
                VecNorm(v, NORM_INFINITY, &ninf);
                if (rank == 0) {
                    std::cout << std::setprecision(15);
                    std::cout << "[NORM] ||" << label << "||_1   = " << n1 << "\n";
                    std::cout << "[NORM] ||" << label << "||_2   = " << n2 << "\n";
                    std::cout << "[NORM] ||" << label << "||_inf = " << ninf << "\n";
                }
            };

            print_vec_norm("b_total", linear_solver_.b().vec());
            print_vec_norm("u", linear_solver_.x().vec());
        }
    }

    dgop_->set_slip(invalid_slip_bc());
    disp_scatter_.begin_scatter(linear_solver_.x(), disp_ghost_);
    disp_scatter_.wait_scatter();
}

void SeasQDOperator::update_traction(BlockView const& state_view) {
    auto disp_view = LocalGhostCompositeView(linear_solver_.x(), disp_ghost_);
    dgop_->set_slip(adapter_->slip_bc(state_view));
    adapter_->traction(disp_view, traction_);
    dgop_->set_slip(invalid_slip_bc());
}

} // namespace tndm
