#include "Elasticity.h"
#include "config.h"
#include "kernels/elasticity/init.h"
#include "kernels/elasticity/kernel.h"
#include "kernels/elasticity/tensor.h"

#include "basis/WarpAndBlend.h"
#include "form/BC.h"
#include "form/DGCurvilinearCommon.h"
#include "form/InverseInequality.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "util/LinearAllocator.h"
#include "util/Stopwatch.h"

#include <Eigen/LU>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>

namespace tensor = tndm::elasticity::tensor;
namespace init = tndm::elasticity::init;
namespace kernel = tndm::elasticity::kernel;

namespace tndm {

namespace {

struct FirstStepDumpConfig {
    bool enabled = false;
    int target_rank = -1;
    std::string output_dir = ".";
    double x_min = -37.5;
    double x_max = -35.5;
    double depth_min = 38.5;
    double depth_max = 40.5;
};

inline FirstStepDumpConfig const& first_step_dump_config() {
    static FirstStepDumpConfig cfg = [] {
        FirstStepDumpConfig c;
        if (auto const* env = std::getenv("TANDEM_FIRST_STEP_DUMP"); env != nullptr) {
            c.enabled = std::string(env) == "1";
        }
        if (auto const* env = std::getenv("TANDEM_FIRST_STEP_TARGET_RANK"); env != nullptr &&
                                                               *env != '\0') {
            c.target_rank = std::atoi(env);
        }
        if (auto const* env = std::getenv("TANDEM_FIRST_STEP_OUTPUT_DIR");
            env != nullptr && *env != '\0') {
            c.output_dir = env;
        }
        return c;
    }();
    return cfg;
}

inline int first_step_dump_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline bool first_step_dump_active() {
    auto const& cfg = first_step_dump_config();
    if (!(cfg.enabled && Elasticity::diag_t_ > 0.0 &&
          (cfg.target_rank < 0 || first_step_dump_rank() == cfg.target_rank))) {
        return false;
    }
    static bool armed_printed = false;
    if (!armed_printed) {
        armed_printed = true;
        std::cerr << "[TND-FS-ARMED] r=" << first_step_dump_rank()
                  << " t=" << Elasticity::diag_t_
                  << " outdir=" << cfg.output_dir << "\n";
    }
    static double first_dump_time = -1.0;
    if (first_dump_time < 0.0) {
        first_dump_time = Elasticity::diag_t_;
    }
    double scale = std::max(1.0, std::abs(first_dump_time));
    return std::abs(Elasticity::diag_t_ - first_dump_time) <= 1e-12 * scale;
}

inline std::string first_step_dump_path(char const* stem) {
    auto const& cfg = first_step_dump_config();
    std::ostringstream oss;
    oss << cfg.output_dir;
    if (!cfg.output_dir.empty() && cfg.output_dir.back() != '/') {
        oss << "/";
    }
    oss << stem << "_r" << first_step_dump_rank() << ".csv";
    return oss.str();
}

template <class CoordsTensor>
bool first_step_dump_match(CoordsTensor const& coords_q) {
    auto const& cfg = first_step_dump_config();
    double cx = 0.0;
    double depth = 0.0;
    auto nq = coords_q.shape(1);
    for (std::size_t q = 0; q < nq; ++q) {
        cx += coords_q(0, q);
        depth += -coords_q(2, q);
    }
    cx /= static_cast<double>(nq);
    depth /= static_cast<double>(nq);
    return (cx >= cfg.x_min && cx <= cfg.x_max && depth >= cfg.depth_min &&
            depth <= cfg.depth_max);
}

inline char const* first_step_face_kind(BC bc, bool is_boundary = false) {
    switch (bc) {
    case BC::Fault:
        return is_boundary ? "fault_boundary" : "fault_skeleton";
    case BC::Dirichlet:
        return is_boundary ? "dirichlet_boundary" : "dirichlet_skeleton";
    default:
        return "other";
    }
}

/// Truncate a dump file on first open, then append for subsequent faces.
/// Prevents stale data from previous runs accumulating in append mode.
inline std::ofstream first_step_open(char const* stem, bool& header_written) {
    if (!header_written) {
        // First open: truncate to discard any leftover from previous runs
        return std::ofstream(first_step_dump_path(stem), std::ios::trunc);
    }
    return std::ofstream(first_step_dump_path(stem), std::ios::app);
}

inline void first_step_write_face_rhs(std::size_t fctNo, FacetInfo const& info,
                                      Matrix<double const> const& coords_q,
                                      Matrix<double const> const& f_q,
                                      Vector<double> const& B0,
                                      Vector<double> const& B1,
                                      bool is_boundary = false) {
    static bool header_written = false;
    std::ofstream out = first_step_open("first_step_face_rhs", header_written);
    if (!out) {
        return;
    }
    if (!header_written) {
        out << "time,face_kind,fct,elem0,elem1,record,index0,index1,value\n";
        header_written = true;
    }
    auto write_row = [&](char const* record, int i0, int i1, double value) {
        out << std::setprecision(17) << Elasticity::diag_t_ << ","
            << first_step_face_kind(info.bc, is_boundary) << "," << fctNo << "," << info.up[0] << ","
            << info.up[1] << "," << record << "," << i0 << "," << i1 << "," << value << "\n";
    };
    for (std::size_t q = 0; q < coords_q.shape(1); ++q) {
        write_row("phys_y", static_cast<int>(q), -1, coords_q(1, q));
        for (std::size_t c = 0; c < f_q.shape(0); ++c) {
            write_row("input_qp", static_cast<int>(q), static_cast<int>(c), f_q(c, q));
        }
    }
    for (std::size_t i = 0; i < B0.size(); ++i) {
        write_row("elvec1", static_cast<int>(i), -1, B0(i));
    }
    for (std::size_t i = 0; i < B1.size(); ++i) {
        write_row("elvec2", static_cast<int>(i), -1, B1(i));
    }
}

inline void first_step_write_face_jump_and_trac(
    std::size_t fctNo, FacetInfo const& info, Matrix<double const> const& coords_q,
    Matrix<double const> const& f_q, Matrix<double const> const& traction_q,
    Matrix<double const> const& u0_q, Matrix<double const> const& u1_q,
    std::vector<double> const& nl_q) {
    static bool jump_header_written = false;
    static bool trac_header_written = false;

    {
        std::ofstream out = first_step_open("first_step_face_jump", jump_header_written);
        if (out) {
            if (!jump_header_written) {
                out << "time,face_kind,fct,elem0,elem1,record,q,component,value\n";
                jump_header_written = true;
            }
            auto write_row = [&](char const* record, int q, int c, double value) {
                out << std::setprecision(17) << Elasticity::diag_t_ << ","
                    << first_step_face_kind(info.bc) << "," << fctNo << "," << info.up[0] << ","
                    << info.up[1] << "," << record << "," << q << "," << c << "," << value
                    << "\n";
            };
            for (std::size_t q = 0; q < traction_q.shape(1); ++q) {
                for (std::size_t c = 0; c < traction_q.shape(0); ++c) {
                    write_row("u1_q", static_cast<int>(q), static_cast<int>(c), u0_q(c, q));
                    write_row("u2_q", static_cast<int>(q), static_cast<int>(c), u1_q(c, q));
                    write_row("slip_q", static_cast<int>(q), static_cast<int>(c), f_q(c, q));
                    write_row("jump_minus_slip", static_cast<int>(q), static_cast<int>(c),
                              u0_q(c, q) - u1_q(c, q) - f_q(c, q));
                }
            }
        }
    }

    {
        std::ofstream out = first_step_open("first_step_face_trac", trac_header_written);
        if (out) {
            if (!trac_header_written) {
                out << "time,face_kind,fct,elem0,elem1,record,index0,index1,value\n";
                trac_header_written = true;
            }
            auto write_row = [&](char const* record, int i0, int i1, double value) {
                out << std::setprecision(17) << Elasticity::diag_t_ << ","
                    << first_step_face_kind(info.bc) << "," << fctNo << "," << info.up[0] << ","
                    << info.up[1] << "," << record << "," << i0 << "," << i1 << "," << value
                    << "\n";
            };
            for (std::size_t q = 0; q < traction_q.shape(1); ++q) {
                write_row("phys_x", static_cast<int>(q), -1, coords_q(0, q));
                write_row("phys_y", static_cast<int>(q), -1, coords_q(1, q));
                write_row("phys_z", static_cast<int>(q), -1, coords_q(2, q));
                write_row("nl_q", static_cast<int>(q), -1, nl_q[q]);
                for (std::size_t c = 0; c < traction_q.shape(0); ++c) {
                    write_row("traction_q", static_cast<int>(q), static_cast<int>(c),
                              traction_q(c, q));
                }
            }
        }
    }
}

} // namespace

Elasticity::Elasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> lam,
                       functional_t<1> mu, std::optional<functional_t<1>> rho, DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      fun_lam(make_volume_functional(std::move(lam))),
      fun_mu(make_volume_functional(std::move(mu))),
      fun_rho(rho ? make_volume_functional(std::move(*rho)) : one_volume_function) {

    MhatInv = space_.inverseMassMatrix();
    E_Q = space_.evaluateBasisAt(volRule.points());
    E_Q_T = space_.evaluateBasisAt(volRule.points(), {1, 0});
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    Dxi_Q_120 = space_.evaluateGradientAt(volRule.points(), {1, 2, 0});

    negative_E_Q_T = Managed<Matrix<double>>(E_Q_T.shape(), std::size_t{ALIGNMENT});
    EigenMap(negative_E_Q_T) = -EigenMap(E_Q_T);

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        Dxi_q_120.emplace_back(space_.evaluateGradientAt(points, {1, 2, 0}));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));

        negative_E_q.emplace_back(space_.evaluateBasisAt(points));
        auto neg = EigenMap(negative_E_q.back());
        neg = -neg;

        negative_E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        auto negT = EigenMap(negative_E_q_T.back());
        negT = -negT;
    }

    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});
}

void Elasticity::compute_mass_matrix(std::size_t elNo, double* M) const {
    kernel::massMatrix mm;
    mm.E_Q = E_Q.data();
    mm.J = vol[elNo].get<AbsDetJ>().data();
    mm.M = M;
    mm.W = volRule.weights().data();
    mm.execute();
}

void Elasticity::compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const {
    compute_mass_matrix(elNo, Minv);

    using MMat = Eigen::Matrix<double, tensor::M::Shape[0], tensor::M::Shape[1]>;
    using MMap = Eigen::Map<MMat, Eigen::Unaligned,
                            Eigen::OuterStride<init::M::Stop[0] - init::M::Start[0]>>;
    auto Minv_eigen = MMap(Minv);
    auto Minv_lu = Eigen::FullPivLU<MMat>(Minv_eigen);
    Minv_eigen = Minv_lu.inverse();
}

void Elasticity::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                   std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());

    volPre.setStorage(std::make_shared<vol_pre_t>(numElements * volRule.size()), 0u, numElements,
                      volRule.size());

    fctPre.setStorage(std::make_shared<fct_pre_t>(numLocalFacets * fctRule.size()), 0u,
                      numLocalFacets, fctRule.size());

    penalty_.resize(numLocalFacets);
    cfl_dt_.resize(numLocalElements, 0.0);
}

void Elasticity::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    alignas(ALIGNMENT) double lam_Q_raw[tensor::lam_Q::size()];
    auto lam_Q = Matrix<double>(lam_Q_raw, 1, volRule.size());
    fun_lam(elNo, lam_Q);

    alignas(ALIGNMENT) double mu_Q_raw[tensor::mu_Q::size()];
    auto mu_Q = Matrix<double>(mu_Q_raw, 1, volRule.size());
    fun_mu(elNo, mu_Q);

    alignas(ALIGNMENT) double rhoInv_Q_raw[tensor::rhoInv_Q::size()];
    auto rhoInv_Q = Matrix<double>(rhoInv_Q_raw, 1, volRule.size());
    fun_rho(elNo, rhoInv_Q);
    for (unsigned q = 0; q < tensor::rhoInv_Q::Shape[0]; ++q) {
        rhoInv_Q(0, q) = 1.0 / rhoInv_Q(0, q);
    }

    alignas(ALIGNMENT) double Mmem[tensor::matM::size()];
    kernel::project_material_lhs krnl_lhs;
    krnl_lhs.matE_Q_T = matE_Q_T.data();
    krnl_lhs.J = vol[elNo].get<AbsDetJ>().data();
    krnl_lhs.matM = Mmem;
    krnl_lhs.W = volRule.weights().data();
    krnl_lhs.execute();

    auto lam_field = material[elNo].get<lam>().data();
    auto mu_field = material[elNo].get<mu>().data();
    auto rhoInv_field = material[elNo].get<rhoInv>().data();
    kernel::project_material_rhs krnl_rhs;
    krnl_rhs.matE_Q_T = matE_Q_T.data();
    krnl_rhs.J = vol[elNo].get<AbsDetJ>().data();
    krnl_rhs.lam = lam_field;
    krnl_rhs.lam_Q = lam_Q_raw;
    krnl_rhs.mu = mu_field;
    krnl_rhs.mu_Q = mu_Q_raw;
    krnl_rhs.rhoInv = rhoInv_field;
    krnl_rhs.rhoInv_Q = rhoInv_Q_raw;
    krnl_rhs.W = volRule.weights().data();
    krnl_rhs.execute();

    using MMap = Eigen::Map<Eigen::Matrix<double, tensor::matM::Shape[0], tensor::matM::Shape[1]>,
                            Eigen::Unaligned,
                            Eigen::OuterStride<init::matM::Stop[0] - init::matM::Start[0]>>;
    using LamMap = Eigen::Map<Eigen::Matrix<double, tensor::lam::Shape[0], 1>, Eigen::Unaligned,
                              Eigen::InnerStride<1>>;
    using MuMap = Eigen::Map<Eigen::Matrix<double, tensor::mu::Shape[0], 1>, Eigen::Unaligned,
                             Eigen::InnerStride<1>>;
    using RhoInvMap = Eigen::Map<Eigen::Matrix<double, tensor::rhoInv::Shape[0], 1>,
                                 Eigen::Unaligned, Eigen::InnerStride<1>>;

    auto proj = MMap(Mmem).fullPivLu();

    auto lam_eigen = LamMap(lam_field);
    lam_eigen = proj.solve(lam_eigen);

    auto mu_eigen = MuMap(mu_field);
    mu_eigen = proj.solve(mu_eigen);

    auto rhoInv_eigen = RhoInvMap(rhoInv_field);
    rhoInv_eigen = proj.solve(rhoInv_eigen);

    auto G_Q = init::G::view::create(vol[elNo].get<JInv>().data()->data());
    auto G_Q_T = init::G_Q_T::view::create(volPre[elNo].get<JInvT>().data()->data());
    for (std::ptrdiff_t q = 0; q < G_Q.shape(2); ++q) {
        for (std::ptrdiff_t i = 0; i < G_Q.shape(0); ++i) {
            for (std::ptrdiff_t j = 0; j < G_Q.shape(1); ++j) {
                G_Q_T(i, j, q) = G_Q(j, i, q);
            }
        }
    }
}

void Elasticity::transpose_JInv(std::size_t fctNo, int side) {
    const auto G_q = (side == 1) ? fct[fctNo].get<JInv1>() : fct[fctNo].get<JInv0>();
    auto G_q_T = (side == 1) ? fctPre[fctNo].get<JInvT1>() : fctPre[fctNo].get<JInvT0>();
    for (std::size_t q = 0; q < fctRule.size(); ++q) {
        for (std::ptrdiff_t i = 0; i < Dim; ++i) {
            for (std::ptrdiff_t j = 0; j < Dim; ++j) {
                G_q_T[q][i + j * Dim] = G_q[q][j + i * Dim];
            }
        }
    }
};

void Elasticity::prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                                  LinearAllocator<double>& scratch) {
    base::prepare_skeleton(fctNo, info, scratch);

    kernel::precomputeSurface krnl;
    for (unsigned side = 0; side < 2; ++side) {
        krnl.matE_q_T(side) = matE_q_T[info.localNo[side]].data();
    }
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();

    for (unsigned side = 0; side < 2; ++side) {
        krnl.lam = material[info.up[side]].get<lam>().data();
        krnl.mu = material[info.up[side]].get<mu>().data();
        krnl.execute(side);
    }

    transpose_JInv(fctNo, 0);
    transpose_JInv(fctNo, 1);
}

void Elasticity::prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                                  LinearAllocator<double>& scratch) {
    base::prepare_boundary(fctNo, info, scratch);

    kernel::precomputeSurface krnl;
    krnl.matE_q_T(0) = matE_q_T[info.localNo[0]].data();
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.lam = material[info.up[0]].get<lam>().data();
    krnl.mu = material[info.up[0]].get<mu>().data();
    krnl.execute(0);

    transpose_JInv(fctNo, 0);
}

void Elasticity::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto lam_field = material[elNo].get<lam>();
    auto mu_field = material[elNo].get<mu>();
    auto rhoInv_field = material[elNo].get<rhoInv>();

    auto J_Q = vol[elNo].get<AbsDetJ>();
    alignas(ALIGNMENT) double Jinv_Q[tensor::Jinv_Q::size()] = {};
    for (unsigned q = 0; q < tensor::Jinv_Q::Shape[0]; ++q) {
        Jinv_Q[q] = 1.0 / J_Q[q];
    }

    kernel::precomputeVolume krnl_pre;
    krnl_pre.matE_Q_T = matE_Q_T.data();
    krnl_pre.J = vol[elNo].get<AbsDetJ>().data();
    krnl_pre.Jinv_Q = Jinv_Q;
    krnl_pre.lam = lam_field.data();
    krnl_pre.lam_W_J_Q = volPre[elNo].template get<lam_W_J_Q>().data();
    krnl_pre.mu = mu_field.data();
    krnl_pre.mu_W_J_Q = volPre[elNo].template get<mu_W_J_Q>().data();
    krnl_pre.rhoInv = rhoInv_field.data();
    krnl_pre.negative_rhoInv_W_Jinv_Q =
        volPre[elNo].template get<negative_rhoInv_W_Jinv_Q>().data();
    krnl_pre.W = volRule.weights().data();
    krnl_pre.execute();
}

std::pair<double, double> Elasticity::stiffness_tensor_bounds(std::size_t elNo) const {
    auto lam_field = material[elNo].get<lam>();
    auto mu_field = material[elNo].get<mu>();
    assert(lam_field.size() == mu_field.size());

    double c0 = std::numeric_limits<double>::max();
    double c1 = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0, n = lam_field.size(); i < n; ++i) {
        c0 = std::min(c0, 2.0 * mu_field[i]);
        c1 = std::max(c1, Dim * lam_field[i] + 2.0 * mu_field[i]);
    }
    return {c0, c1};
}

double Elasticity::inverse_density_upper_bound(std::size_t elNo) const {
    auto rhoInv_field = material[elNo].get<rhoInv>();

    double ir1 = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0, n = rhoInv_field.size(); i < n; ++i) {
        ir1 = std::max(ir1, rhoInv_field[i]);
    }
    return ir1;
}

void Elasticity::prepare_penalty(std::size_t fctNo, FacetInfo const& info,
                                 LinearAllocator<double>&) {
    auto const p = [&](int side) {
        const auto [c0, c1] = stiffness_tensor_bounds(info.up[side]);
        constexpr double c_N_1 = InverseInequality<Dim>::trace_constant(PolynomialDegree - 1);
        return (Dim + 1) * c_N_1 * (area_[fctNo] / volume_[info.up[side]]) * (c1 * c1 / c0);
    };

    if (info.up[0] != info.up[1]) {
        penalty_[fctNo] = (p(0) + p(1)) / 4.0;
    } else {
        penalty_[fctNo] = p(0);
    }
}

void Elasticity::prepare_cfl(std::size_t elNo, mneme::span<SideInfo> info,
                             LinearAllocator<double>&) {
    double l_max = 0.0; // Bound for maximum eigenvalue
    double bnd_area = 0.0;
    for (std::size_t f = 0; f < NumFacets; ++f) {
        const bool is_skeleton_face = elNo != info[f].lid;
        const double gamma = is_skeleton_face ? 2.0 : 1.0;
        const auto fctNo = info[f].fctNo;

        constexpr double c_N = InverseInequality<Dim>::trace_constant(PolynomialDegree);
        l_max += gamma * penalty_[fctNo] * c_N * area_[fctNo] / volume_[elNo];
        bnd_area += area_[fctNo];
    }

    const auto [c0, c1] = stiffness_tensor_bounds(elNo);
    const double ir1 = inverse_density_upper_bound(elNo);
    const double h_1 = bnd_area / volume_[elNo];
    constexpr double C_N = InverseInequality<Dim>::grad_constant(PolynomialDegree);
    l_max += c1 * ir1 * C_N * h_1 * h_1;
    l_max *= 2.0;
    cfl_dt_[elNo] = 1.0 / sqrt(l_max);
}

bool Elasticity::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                                 LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(Dxi_Q.shape(0) == tensor::Dxi_Q::Shape[0]);
    assert(Dxi_Q.shape(1) == tensor::Dxi_Q::Shape[1]);
    assert(Dxi_Q.shape(2) == tensor::Dxi_Q::Shape[2]);

    kernel::Dx_Q dxKrnl;
    dxKrnl.Dx_Q = Dx_Q;
    dxKrnl.Dxi_Q = Dxi_Q.data();
    dxKrnl.G = vol[elNo].get<JInv>().data()->data();
    dxKrnl.execute();

    kernel::assembleVolume krnl;
    krnl.A = A00.data();
    krnl.delta = init::delta::Values;
    krnl.Dx_Q = Dx_Q;
    krnl.lam_W_J_Q = volPre[elNo].get<lam_W_J_Q>().data();
    krnl.mu_W_J_Q = volPre[elNo].get<mu_W_J_Q>().data();
    krnl.execute();

    // First-step K dump: volume contribution
    if (first_step_dump_active()) {
        // Check if this element is a parent of any matched face
        // (dump all volume contributions — filter offline)
        static bool vol_header = false;
        std::ofstream out = first_step_open("first_step_K_contributions", vol_header);
        if (out) {
            if (!vol_header) {
                out << "source,elem_or_face,elem1,elem2,block,row,col,value\n";
                vol_header = true;
            }
            int n = static_cast<int>(A00.shape(0));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (std::abs(A00(i,j)) > 1e-50) {
                        out << "volume," << elNo << "," << elNo << ",-1,A00,"
                            << i << "," << j << "," << std::setprecision(17)
                            << A00(i,j) << "\n";
                    }
                }
            }
        }
    }

    return true;
}

bool Elasticity::assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                   Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                                   LinearAllocator<double>& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    alignas(ALIGNMENT) double traction_op_q0[tensor::traction_op_q::size(0)];
    alignas(ALIGNMENT) double traction_op_q1[tensor::traction_op_q::size(1)];

    kernel::assembleTractionOp tOpKrnl;
    tOpKrnl.delta = init::delta::Values;
    tOpKrnl.Dx_q(0) = Dx_q0;
    tOpKrnl.Dx_q(1) = Dx_q1;
    tOpKrnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    tOpKrnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    tOpKrnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    tOpKrnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    tOpKrnl.n_q = fct[fctNo].get<Normal>().data()->data();
    tOpKrnl.traction_op_q(0) = traction_op_q0;
    tOpKrnl.traction_op_q(1) = traction_op_q1;
    tOpKrnl.execute(0);
    tOpKrnl.execute(1);

    alignas(ALIGNMENT) double L_q0[tensor::L_q::size(0)];
    alignas(ALIGNMENT) double L_q1[tensor::L_q::size(1)];
    auto L_q = std::array<double*, 2>{L_q0, L_q1};

    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Lift1[tensor::Lift::size(1)];
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i) {
            compute_inverse_mass_matrix(info.up[i], Minv[i]);
        }

        kernel::lift_skeleton lift;
        lift.delta = init::delta::Values;
        lift.Lift(0) = Lift0;
        lift.Lift(1) = Lift1;
        lift.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
        lift.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
        lift.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
        lift.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.L_q(i) = L_q[i];
            lift.Minv(i) = Minv[i];
        }
        lift.execute(0);
        lift.execute(1);
    } else { // IP
        kernel::lift_ip lift;
        lift.delta = init::delta::Values;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.L_q(i) = L_q[i];
        }
        lift.execute(0);
        lift.execute(1);
    }

    kernel::assembleSurface krnl;
    krnl.c00 = -0.5;
    krnl.c01 = -krnl.c00;
    krnl.c10 = epsilon * 0.5;
    krnl.c11 = -krnl.c10;
    krnl.c20 = penalty(fctNo);
    krnl.c21 = -krnl.c20;
    krnl.a(0, 0) = A00.data();
    krnl.a(0, 1) = A01.data();
    krnl.a(1, 0) = A10.data();
    krnl.a(1, 1) = A11.data();
    for (unsigned side = 0; side < 2; ++side) {
        krnl.E_q(side) = E_q[info.localNo[side]].data();
        krnl.L_q(side) = L_q[side];
    }
    krnl.traction_op_q(0) = traction_op_q0;
    krnl.traction_op_q(1) = traction_op_q1;
    krnl.w = fctRule.weights().data();
    krnl.execute(0, 0);
    krnl.execute(0, 1);
    krnl.execute(1, 0);
    krnl.execute(1, 1);

    // First-step K dump: skeleton face contribution
    if (first_step_dump_active()) {
        auto coords_q = Matrix<double const>(
            fct[fctNo].template get<Coords>().data()->data(),
            NumQuantities, fctRule.size());
        if (first_step_dump_match(coords_q)) {
            static bool skel_header = false;
            std::ofstream out = first_step_open("first_step_K_contributions", skel_header);
            if (out) {
                if (!skel_header) {
                    out << "source,elem_or_face,elem1,elem2,block,row,col,value\n";
                    skel_header = true;
                }
                auto dump_block = [&](char const* blk, Matrix<double> const& M) {
                    for (std::size_t i = 0; i < M.shape(0); i++) {
                        for (std::size_t j = 0; j < M.shape(1); j++) {
                            if (std::abs(M(i,j)) > 1e-50) {
                                out << "skeleton_face," << fctNo << ","
                                    << info.up[0] << "," << info.up[1] << ","
                                    << blk << "," << i << "," << j << ","
                                    << std::setprecision(17) << M(i,j) << "\n";
                            }
                        }
                    }
                };
                dump_block("A00", A00);
                dump_block("A01", A01);
                dump_block("A10", A10);
                dump_block("A11", A11);
            }
        }
    }

    return true;
}

bool Elasticity::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                   LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    alignas(ALIGNMENT) double traction_op_q0[tensor::traction_op_q::size(0)];

    kernel::assembleTractionOp tOpKrnl;
    tOpKrnl.delta = init::delta::Values;
    tOpKrnl.Dx_q(0) = Dx_q0;
    tOpKrnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    tOpKrnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    tOpKrnl.n_q = fct[fctNo].get<Normal>().data()->data();
    tOpKrnl.traction_op_q(0) = traction_op_q0;
    tOpKrnl.execute(0);

    alignas(ALIGNMENT) double L_q0[tensor::L_q::size(0)];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        kernel::lift_boundary lift;
        lift.delta = init::delta::Values;
        lift.Lift(0) = Lift0;
        lift.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
        lift.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.L_q(0) = L_q0;
        lift.Minv(0) = Minv0;
        lift.execute();
    } else { // IP
        kernel::lift_ip lift;
        lift.delta = init::delta::Values;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.L_q(0) = L_q0;
        lift.execute(0);
    }

    kernel::assembleSurface krnl;
    krnl.c00 = -1.0;
    krnl.c10 = epsilon;
    krnl.c20 = penalty(fctNo);
    krnl.a(0, 0) = A00.data();
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.L_q(0) = L_q0;
    krnl.traction_op_q(0) = traction_op_q0;
    krnl.w = fctRule.weights().data();
    krnl.execute(0, 0);

    return true;
}

bool Elasticity::rhs_volume(std::size_t elNo, Vector<double>& B,
                            LinearAllocator<double>& scratch) const {
    if (!fun_force) {
        return false;
    }

    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F_Q_raw[tensor::F_Q::size()];
    assert(tensor::F_Q::Shape[1] == volRule.size());

    auto F_Q = Matrix<double>(F_Q_raw, NumQuantities, volRule.size());
    (*fun_force)(elNo, F_Q);

    kernel::rhsVolume rhs;
    rhs.E_Q = E_Q.data();
    rhs.F_Q = F_Q_raw;
    rhs.J = vol[elNo].get<AbsDetJ>().data();
    rhs.W = volRule.weights().data();
    rhs.b = B.data();
    rhs.execute();
    return true;
}

bool Elasticity::bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault && fun_slip) {
        (*fun_slip)(fctNo, f_q, false);
    } else if (bc == BC::Dirichlet && fun_dirichlet) {
        (*fun_dirichlet)(fctNo, f_q, false);
    } else {
        return false;
    }
    return true;
}
bool Elasticity::bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault && fun_slip) {
        (*fun_slip)(fctNo, f_q, true);
        for (std::size_t q = 0; q < tensor::f_q::Shape[1]; ++q) {
            for (std::size_t p = 0; p < NumQuantities; ++p) {
                f_q(p, q) *= 0.5;
            }
        }
    } else if (bc == BC::Dirichlet && fun_dirichlet) {
        (*fun_dirichlet)(fctNo, f_q, true);
    } else {
        return false;
    }
    return true;
}

bool Elasticity::rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              Vector<double>& B1, LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
        alignas(ALIGNMENT) double f_lifted1[tensor::f_lifted::size(1)];
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i) {
            compute_inverse_mass_matrix(info.up[i], Minv[i]);
        }

        kernel::rhs_lift_skeleton lift;
        lift.delta = init::delta::Values;
        lift.f_q = f_q_raw;
        lift.f_lifted(0) = f_lifted0;
        lift.f_lifted(1) = f_lifted1;
        lift.f_lifted_q = f_lifted_q;
        lift.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
        lift.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
        lift.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
        lift.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.Minv(i) = Minv[i];
        }
        lift.execute();
    } else { // IP
        kernel::rhs_lift_ip lift;
        lift.f_q = f_q_raw;
        lift.f_lifted_q = f_lifted_q;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.execute();
    }

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[1]].data();
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_1>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_1>().data();
    rhs.execute();

    if (first_step_dump_active()) {
        auto coords_q = Matrix<double const>(fct[fctNo].template get<Coords>().data()->data(),
                                             NumQuantities, fctRule.size());
        if ((info.bc == BC::Fault || info.bc == BC::Dirichlet) &&
            first_step_dump_match(coords_q)) {
            auto f_q = Matrix<double const>(f_q_raw, NumQuantities, fctRule.size());
            first_step_write_face_rhs(fctNo, info, coords_q, f_q, B0, B1);
        }
    }

    return true;
}

bool Elasticity::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
        alignas(ALIGNMENT) double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        kernel::rhs_lift_boundary lift;
        lift.delta = init::delta::Values;
        lift.f_q = f_q_raw;
        lift.f_lifted(0) = f_lifted0;
        lift.f_lifted_q = f_lifted_q;
        lift.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
        lift.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.Minv(0) = Minv0;
        lift.execute();
    } else { // IP
        kernel::rhs_lift_ip lift;
        lift.f_q = f_q_raw;
        lift.f_lifted_q = f_lifted_q;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.execute();
    }

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    // First-step dump for boundary faces (mirrors the rhs_skeleton hook).
    // Boundary faces only have B0 (one parent element), so pass an empty B1.
    if (first_step_dump_active()) {
        auto coords_q = Matrix<double const>(fct[fctNo].template get<Coords>().data()->data(),
                                             NumQuantities, fctRule.size());
        if (info.bc == BC::Dirichlet && first_step_dump_match(coords_q)) {
            auto f_q = Matrix<double const>(f_q_raw, NumQuantities, fctRule.size());
            // Create a zero-sized B1 placeholder (boundary has no second element)
            Vector<double> B1_empty;
            first_step_write_face_rhs(fctNo, info, coords_q, f_q, B0, B1_empty,
                                      true /* is_boundary */);
        }
    }

    return true;
}

template <bool WithRHS>
void Elasticity::apply_(std::size_t elNo, mneme::span<SideInfo> info,
                        Vector<double const> const& x_0,
                        std::array<Vector<double const>, NumFacets> const& x_n,
                        Vector<double>& y_0) const {
    alignas(ALIGNMENT) double Ju_Q[tensor::Ju_Q::size()];
    kernel::apply_volume av;
    av.delta = init::delta::Values;
    av.Dxi_Q = Dxi_Q.data();
    av.Dxi_Q_120 = Dxi_Q_120.data();
    av.Ju_Q = Ju_Q;
    av.G = vol[elNo].get<JInv>().data()->data();
    av.G_Q_T = volPre[elNo].get<JInvT>().data()->data();
    av.lam_W_J_Q = volPre[elNo].get<lam_W_J_Q>().data();
    av.mu_W_J_Q = volPre[elNo].get<mu_W_J_Q>().data();
    av.U = x_0.data();
    av.Unew = y_0.data();
    av.execute();

    alignas(ALIGNMENT) double Ju_q0[tensor::Ju_q::size(0)];
    alignas(ALIGNMENT) double Ju_q1[tensor::Ju_q::size(1)];
    alignas(ALIGNMENT) double n_q_flipped[tensor::n_q::size()];
    alignas(ALIGNMENT) double n_unit_q_flipped[tensor::n_unit_q::size()];
    for (std::size_t f = 0; f < NumFacets; ++f) {
        bool is_skeleton_face = elNo != info[f].lid;
        bool is_fault_or_dirichlet = info[f].bc == BC::Fault || info[f].bc == BC::Dirichlet;

        auto fctNo = info[f].fctNo;
        double const* n_q = fct[fctNo].get<Normal>().data()->data();
        double const* n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
        double const* lam_q0 = fctPre[fctNo].get<lam_q_0>().data();
        double const* lam_q1 = fctPre[fctNo].get<lam_q_1>().data();
        double const* mu_q0 = fctPre[fctNo].get<mu_q_0>().data();
        double const* mu_q1 = fctPre[fctNo].get<mu_q_1>().data();
        double const* G_q_T0 = fctPre[fctNo].get<JInvT0>().data()->data();
        double const* G_q_T1 = fctPre[fctNo].get<JInvT1>().data()->data();
        if (is_skeleton_face && info[f].side == 1) {
            std::swap(lam_q0, lam_q1);
            std::swap(mu_q0, mu_q1);
            std::swap(G_q_T0, G_q_T1);

            for (int i = 0; i < tensor::n_q::size(); ++i) {
                n_q_flipped[i] = -n_q[i];
            }
            n_q = n_q_flipped;
            for (int i = 0; i < tensor::n_unit_q::size(); ++i) {
                n_unit_q_flipped[i] = -n_unit_q[i];
            }
            n_unit_q = n_unit_q_flipped;
        }

        alignas(ALIGNMENT) double u_hat_minus_u_q[tensor::u_hat_minus_u_q::size()] = {};
        alignas(ALIGNMENT) double sigma_hat_q[tensor::sigma_hat_q::size()] = {};

        if (info[f].bc == BC::None || (is_skeleton_face && is_fault_or_dirichlet)) {
            kernel::flux_u_skeleton fu;
            fu.negative_E_q_T(0) = negative_E_q_T[f].data();
            fu.E_q_T(1) = E_q_T[info[f].localNo].data();
            fu.U = x_0.data();
            fu.U_ext = x_n[f].data();
            fu.u_hat_minus_u_q = u_hat_minus_u_q;
            fu.execute();

            kernel::flux_sigma_skeleton fs;
            fs.c00 = -penalty(fctNo);
            fs.delta = init::delta::Values;
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.Dxi_q_120(1) = Dxi_q_120[info[f].localNo].data();
            fs.E_q_T(0) = E_q_T[f].data();
            fs.G_q_T(0) = G_q_T0;
            fs.G_q_T(1) = G_q_T1;
            fs.lam_q(0) = lam_q0;
            fs.lam_q(1) = lam_q1;
            fs.mu_q(0) = mu_q0;
            fs.mu_q(1) = mu_q1;
            fs.negative_E_q_T(1) = negative_E_q_T[info[f].localNo].data();
            fs.U = x_0.data();
            fs.U_ext = x_n[f].data();
            fs.n_unit_q = n_unit_q;
            fs.sigma_hat_q = sigma_hat_q;
            fs.Ju_q(0) = Ju_q0;
            fs.Ju_q(1) = Ju_q1;
            fs.execute();

            if constexpr (WithRHS) {
                alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
                if (bc_skeleton(fctNo, info[f].bc, f_q_raw)) {
                    double sign = info[f].side == 1 ? -1.0 : 1.0;
                    kernel::flux_u_add_bc fub;
                    fub.c00 = 0.5 * sign;
                    fub.f_q = f_q_raw;
                    fub.u_hat_minus_u_q = u_hat_minus_u_q;
                    fub.execute();

                    kernel::flux_sigma_add_bc fsb;
                    fsb.c00 = sign * penalty(fctNo);
                    fsb.f_q = f_q_raw;
                    fsb.n_unit_q = n_unit_q;
                    fsb.sigma_hat_q = sigma_hat_q;
                    fsb.execute();
                }
            }
        } else if (is_fault_or_dirichlet) {
            kernel::flux_u_boundary fu;
            fu.U = x_0.data();
            fu.u_hat_minus_u_q = u_hat_minus_u_q;
            fu.negative_E_q_T(0) = negative_E_q_T[f].data();
            fu.execute();

            kernel::flux_sigma_boundary fs;
            fs.c00 = -penalty(fctNo);
            fs.delta = init::delta::Values;
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.G_q_T(0) = G_q_T0;
            fs.E_q_T(0) = E_q_T[f].data();
            fs.lam_q(0) = lam_q0;
            fs.mu_q(0) = mu_q0;
            fs.U = x_0.data();
            fs.n_unit_q = n_unit_q;
            fs.sigma_hat_q = sigma_hat_q;
            fs.Ju_q(0) = Ju_q0;
            fs.execute();

            if constexpr (WithRHS) {
                alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
                if (bc_boundary(fctNo, info[f].bc, f_q_raw)) {
                    kernel::flux_u_add_bc fub;
                    fub.c00 = 1.0;
                    fub.f_q = f_q_raw;
                    fub.u_hat_minus_u_q = u_hat_minus_u_q;
                    fub.execute();

                    kernel::flux_sigma_add_bc fsb;
                    fsb.c00 = penalty(fctNo);
                    fsb.f_q = f_q_raw;
                    fsb.n_unit_q = n_unit_q;
                    fsb.sigma_hat_q = sigma_hat_q;
                    fsb.execute();
                }
            }
        } else {
            continue;
        }

        kernel::apply_facet af;
        af.delta = init::delta::Values;
        af.Dxi_q(0) = Dxi_q[f].data();
        af.negative_E_q(0) = negative_E_q[f].data();
        af.G_q_T(0) = G_q_T0;
        af.lam_q(0) = lam_q0;
        af.mu_q(0) = mu_q0;
        af.n_q = n_q;
        af.sigma_hat_q = sigma_hat_q;
        af.u_hat_minus_u_q = u_hat_minus_u_q;
        af.Unew = y_0.data();
        af.w = fctRule.weights().data();
        af.execute();
    }
}

void Elasticity::apply(std::size_t elNo, mneme::span<SideInfo> info,
                       Vector<double const> const& x_0,
                       std::array<Vector<double const>, NumFacets> const& x_n,
                       Vector<double>& y_0) const {
    apply_<false>(elNo, std::move(info), x_0, x_n, y_0);
}

void Elasticity::wave_rhs(std::size_t elNo, mneme::span<SideInfo> info,
                          Vector<double const> const& x_0,
                          std::array<Vector<double const>, NumFacets> const& x_n,
                          Vector<double>& y_0) const {
    alignas(ALIGNMENT) double rhs_raw[tensor::Unew::size()];
    auto rhs = Vector<double>(rhs_raw, y_0.shape(0));
    apply_<true>(elNo, std::move(info), x_0, x_n, rhs);

    kernel::apply_inverse_mass krnl;
    krnl.E_Q = E_Q.data();
    krnl.MinvRef = MhatInv.data();
    krnl.Jinv_Q = volPre[elNo].get<negative_rhoInv_W_Jinv_Q>().data();
    krnl.U = rhs_raw;
    krnl.Unew = y_0.data();
    krnl.execute();
}

void Elasticity::project(std::size_t elNo, volume_functional_t x, Vector<double>& y) const {
    alignas(ALIGNMENT) double U_Q_raw[tensor::U_Q::size()];
    alignas(ALIGNMENT) double U_raw[tensor::U::size()];

    auto U_Q = Matrix<double>(U_Q_raw, NumQuantities, volRule.size());
    x(elNo, U_Q);

    kernel::project_u_rhs krnl;
    krnl.E_Q = E_Q.data();
    krnl.J = vol[elNo].get<AbsDetJ>().data();
    krnl.U = U_raw;
    krnl.U_Q = U_Q_raw;
    krnl.W = volRule.weights().data();
    krnl.execute();

    auto J_Q = vol[elNo].get<AbsDetJ>();
    auto const& w = volRule.weights();
    alignas(ALIGNMENT) double Jinv_Q[tensor::Jinv_Q::size()] = {};
    for (unsigned q = 0; q < tensor::Jinv_Q::Shape[0]; ++q) {
        Jinv_Q[q] = w[q] / J_Q[q];
    }

    kernel::apply_inverse_mass im;
    im.E_Q = E_Q.data();
    im.MinvRef = MhatInv.data();
    im.Jinv_Q = Jinv_Q;
    im.U = U_raw;
    im.Unew = y.data();
    im.execute();
}

std::size_t Elasticity::flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const {
    std::size_t flops = kernel::apply_volume::HardwareFlops;
    for (std::size_t f = 0; f < NumFacets; ++f) {
        bool is_skeleton_face = elNo != info[f].lid;
        bool is_fault_or_dirichlet = info[f].bc == BC::Fault || info[f].bc == BC::Dirichlet;
        if (info[f].bc == BC::None || (is_skeleton_face && is_fault_or_dirichlet)) {
            flops += kernel::flux_u_skeleton::HardwareFlops;
            flops += kernel::flux_sigma_skeleton::HardwareFlops;
        } else if (is_fault_or_dirichlet) {
            flops += kernel::flux_u_boundary::HardwareFlops;
            flops += kernel::flux_sigma_boundary::HardwareFlops;
        } else {
            continue;
        }
        flops += kernel::apply_facet::HardwareFlops;
    }
    return flops;
}

void Elasticity::coefficients_volume(std::size_t elNo, Matrix<double>& C,
                                     LinearAllocator<double>&) const {
    auto const coeff_lam = material[elNo].get<lam>();
    auto const coeff_mu = material[elNo].get<mu>();
    assert(coeff_lam.size() == C.shape(0));
    assert(coeff_mu.size() == C.shape(0));
    assert(2 == C.shape(1));
    for (std::size_t i = 0; i < C.shape(0); ++i) {
        C(i, 0) = coeff_lam[i];
        C(i, 1) = coeff_mu[i];
    }
}

TensorBase<Matrix<double>> Elasticity::tractionResultInfo() const {
    return TensorBase<Matrix<double>>(tensor::traction_q::Shape[0], tensor::traction_q::Shape[1]);
}

void Elasticity::traction_skeleton(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Vector<double const>& u1,
                                   Matrix<double>& result) const {
    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));
    assert(u1.size() == tensor::u::size(1));

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    kernel::compute_traction krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.Dx_q(0) = Dx_q0;
    krnl.Dx_q(1) = Dx_q1;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();

    if (first_step_dump_active()) {
        auto coords_q = Matrix<double const>(fct[fctNo].template get<Coords>().data()->data(),
                                             NumQuantities, fctRule.size());
        if ((info.bc == BC::Fault || info.bc == BC::Dirichlet) &&
            first_step_dump_match(coords_q)) {
            auto f_q = Matrix<double const>(f_q_raw, NumQuantities, fctRule.size());
            auto traction_q = Matrix<double const>(result.data(), NumQuantities, fctRule.size());
            std::vector<double> u0_q_raw(NumQuantities * fctRule.size(), 0.0);
            std::vector<double> u1_q_raw(NumQuantities * fctRule.size(), 0.0);
            auto u0_q = Matrix<double>(u0_q_raw.data(), NumQuantities, fctRule.size());
            auto u1_q = Matrix<double>(u1_q_raw.data(), NumQuantities, fctRule.size());
            auto const* E_q0 = E_q[info.localNo[0]].data();
            auto const* E_q1 = E_q[info.localNo[1]].data();
            std::size_t nbf_loc = E_q[info.localNo[0]].shape(0);
            // Sanity check: partition of unity (sum_l E_q[l,q] should be 1)
            for (std::size_t q = 0; q < fctRule.size(); ++q) {
                double pou0 = 0.0, pou1 = 0.0;
                for (std::size_t l = 0; l < nbf_loc; ++l) {
                    pou0 += E_q0[l * fctRule.size() + q];
                    pou1 += E_q1[l * fctRule.size() + q];
                }
                if (std::abs(pou0 - 1.0) > 1e-10 || std::abs(pou1 - 1.0) > 1e-10) {
                    std::cerr << "[TND-FS-WARN] E_q partition-of-unity failure at fct="
                              << fctNo << " q=" << q
                              << " pou0=" << pou0 << " pou1=" << pou1 << "\n";
                }
            }
            for (std::size_t q = 0; q < fctRule.size(); ++q) {
                for (std::size_t c = 0; c < NumQuantities; ++c) {
                    double v0 = 0.0;
                    double v1 = 0.0;
                    for (std::size_t l = 0; l < nbf_loc; ++l) {
                        v0 += E_q0[l * fctRule.size() + q] * u0.data()[l * NumQuantities + c];
                        v1 += E_q1[l * fctRule.size() + q] * u1.data()[l * NumQuantities + c];
                    }
                    u0_q(c, q) = v0;
                    u1_q(c, q) = v1;
                }
            }
            std::vector<double> nl_q(fctRule.size());
            auto unit_normal_q =
                Tensor(fct[fctNo].template get<UnitNormal>().data()->data(),
                       cl_->normalResultInfo(fctRule.size()));
            auto normal_q = Tensor(fct[fctNo].template get<Normal>().data()->data(),
                                   cl_->normalResultInfo(fctRule.size()));
            for (std::size_t q = 0; q < fctRule.size(); ++q) {
                double nl = 0.0;
                for (std::size_t d = 0; d < NumQuantities; ++d) {
                    nl += normal_q(d, q) * unit_normal_q(d, q);
                }
                nl_q[q] = nl;
            }
            auto u0_q_c = Matrix<double const>(u0_q.data(), u0_q.shape(0), u0_q.shape(1));
            auto u1_q_c = Matrix<double const>(u1_q.data(), u1_q.shape(0), u1_q.shape(1));
            first_step_write_face_jump_and_trac(fctNo, info, coords_q, f_q, traction_q, u0_q_c,
                                                u1_q_c, nl_q);
        }
    }

    // v58: one-shot dump of traction_q at the left shallow tip face.
    // Opt-in: set TANDEM_DIAG_TIP_UY=1 to enable.
    // Matches the same physical neighborhood targeted by MFEM [TIP-UY].
    {
        static int tq_enabled = -1;
        static bool tq_armed_printed = false;
        if (tq_enabled < 0) {
            const char* env = std::getenv("TANDEM_DIAG_TIP_UY");
            tq_enabled = (env && std::string(env) == "1") ? 1 : 0;
        }
        static bool tq_done = false;
        static int tq_dbg_count = 0;
        if (tq_enabled && !tq_done) {
            if (!tq_armed_printed) {
                tq_armed_printed = true;
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                std::cerr << "[TND-TQ-ARMED] r=" << rank
                          << " fct=" << fctNo
                          << " up0=" << info.up[0]
                          << " up1=" << info.up[1]
                          << "\n";
            }
            auto coords_q = Tensor(fct[fctNo].template get<Coords>().data()->data(),
                                   cl_->mapResultInfo(fctRule.size()));
            auto unit_normal_q = Tensor(fct[fctNo].template get<UnitNormal>().data()->data(),
                                        cl_->normalResultInfo(fctRule.size()));
            int nq = result.shape(1);
            double Ty0 = result(1, 0);
            double Tx0 = result(0, 0);
            double Tz0 = result(2, 0);
            double cx = 0.0;
            double cz = 0.0;
            double ny_avg = 0.0;
            for (int q = 0; q < nq; q++) {
                cx += coords_q(0, q);
                cz += -coords_q(2, q);
                ny_avg += unit_normal_q(1, q);
            }
            cx /= nq;
            cz /= nq;
            ny_avg /= nq;

            if (tq_dbg_count < 3) {
                tq_dbg_count++;
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                std::cerr << "[TND-TQ-DBG] r=" << rank
                          << " fct=" << fctNo
                          << " call=" << tq_dbg_count
                          << " Tx0=" << Tx0
                          << " Ty0=" << Ty0
                          << " Tz0=" << Tz0
                          << " cx=" << cx
                          << " cz=" << cz
                          << " ny_avg=" << ny_avg
                          << " up0=" << info.up[0]
                          << " up1=" << info.up[1]
                          << "\n";
            }

            // Target MFEM face: cx=-49.4401, cz=2.0121
            bool left_tip_match =
                (std::abs(cx + 49.44) < 0.3 &&
                 std::abs(cz - 2.01) < 0.3 &&
                 std::abs(std::abs(ny_avg) - 1.0) < 0.1);

            if (left_tip_match && std::abs(Ty0) > 1e-30) {
                tq_done = true;
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                double pen = penalty_[fctNo];
                double c00 = -pen;
                double vol0 = volume_[info.up[0]];
                double vol1 = volume_[info.up[1]];
                double area = area_[fctNo];

                // Buffer all output, flush once to avoid MPI interleaving
                std::ostringstream buf;
                buf << std::scientific << std::setprecision(10)
                    << "[TND-TQ] t=" << diag_t_
                    << " r=" << rank
                    << " fct=" << fctNo
                    << " key=(" << info.key[0]
                    << "," << info.key[1]
                    << "," << info.key[2] << ")"
                    << " pen=" << pen
                    << " area=" << area
                    << " vol0=" << vol0 << " vol1=" << vol1
                    << "\n";

                auto const* E_q0 = E_q[info.localNo[0]].data();
                auto const* E_q1 = E_q[info.localNo[1]].data();
                std::size_t nbf_loc = E_q[info.localNo[0]].shape(0);

                for (int q = 0; q < nq; q++) {
                    double u0_y = 0.0, u1_y = 0.0;
                    auto const* u0d = u0.data();
                    auto const* u1d = u1.data();
                    for (std::size_t l = 0; l < nbf_loc; l++) {
                        u0_y += E_q0[l * nq + q] * u0d[l * Dim + 1];
                        u1_y += E_q1[l * nq + q] * u1d[l * Dim + 1];
                    }
                    double slip_y = f_q_raw[1 * nq + q];
                    double jump_y = u0_y - u1_y - slip_y;

                    double Ty_penalty = c00 * jump_y;
                    double Ty = result(1, q);
                    double Ty_stress = Ty - Ty_penalty;

                    buf << std::scientific << std::setprecision(10)
                        << "[TND-TQ] q=" << q
                        << " xyz=(" << coords_q(0, q)
                        << "," << coords_q(1, q)
                        << "," << coords_q(2, q) << ")"
                        << " ny=" << unit_normal_q(1, q)
                        << " u0_y=" << u0_y
                        << " u1_y=" << u1_y
                        << " slip_y=" << slip_y
                        << " jump_y=" << jump_y
                        << " Ty_stress=" << Ty_stress
                        << " Ty_penalty=" << Ty_penalty
                        << " Ty=" << Ty
                        << "\n";
                }
                std::cerr << buf.str() << std::flush;
            }
        }
    }
}

void Elasticity::traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Matrix<double>& result) const {

    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    kernel::compute_traction_bnd krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.Dx_q(0) = Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.execute();
}

} // namespace tndm
