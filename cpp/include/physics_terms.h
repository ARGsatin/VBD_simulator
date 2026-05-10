#pragma once

#include "types.h"
#include <Eigen/Dense>

namespace vbd {

// 组装局部物理项
void build_local_physics_terms(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x_prev,
    LocalPhysicsTerms& out);

} // namespace vbd
