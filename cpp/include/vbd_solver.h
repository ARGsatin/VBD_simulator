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

// solve_lift_and_relax：单步提升 + 单次静平衡（控制反转架构）
// Python 侧接管时间流逝循环，反复调用此单步函数
VBDSolveResult solve_lift_and_relax(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id,
    const std::vector<int>& lifting_top);

} // namespace vbd
