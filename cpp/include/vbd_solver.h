#pragma once

#include "types.h"
#include <vector>

namespace vbd {

// ============================================================================
// 核心求解入口
// ============================================================================

// solve_until_stable：隐式 VBD 迭代直到静力平衡
VBDSolveResult solve_until_stable(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id);

// solve_with_lift：带平台提升剥离 + 静平衡
VBDSolveResult solve_with_lift(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id,
    const std::vector<int>& lifting_top);

} // namespace vbd
