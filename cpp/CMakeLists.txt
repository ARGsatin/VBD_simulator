cmake_minimum_required(VERSION 3.18)
project(vbd_solver_cpp LANGUAGES CXX)

# ── C++ 标准 ──
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenMP ──
find_package(OpenMP REQUIRED)

# ── 优化标志 ──
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DNDEBUG /openmp")
endif()

# ── 拉取 Eigen ──
include(FetchContent)
FetchContent_Declare(
    Eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen)

# ── pybind11（优先用 pip 安装的，其次用国内镜像）──
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 未通过 find_package 找到，从 Gitee 镜像下载 …")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://gitee.com/mirrors/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── 头文件目录 ──
set(CPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── 共享源文件 ──
set(VBD_SOURCES
    src/elastic_energy.cpp
    src/physics_terms.cpp
    src/vbd_solver.cpp
)
set(VBD_HEADERS
    include/types.h
    include/vbd_solver.h
    include/elastic_energy.h
    include/physics_terms.h
)

# ═══════════════════════════════════════════════════════════════════════
# 标准目标（OpenMP 多线程，用于 CLI/批处理）
# ═══════════════════════════════════════════════════════════════════════
pybind11_add_module(hydrogel_vbd_cpp ${VBD_SOURCES} bindings/pybind_vbd.cpp)
target_include_directories(hydrogel_vbd_cpp PRIVATE ${CPP_INCLUDE_DIR})
target_link_libraries(hydrogel_vbd_cpp PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
install(TARGETS hydrogel_vbd_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# ═══════════════════════════════════════════════════════════════════════
# Qt/GUI 安全目标（无 OpenMP，避免 QThread + vcomp.dll segfault）
# ═══════════════════════════════════════════════════════════════════════
pybind11_add_module(hydrogel_vbd_cpp_qt ${VBD_SOURCES} bindings/pybind_vbd.cpp)
target_include_directories(hydrogel_vbd_cpp_qt PRIVATE ${CPP_INCLUDE_DIR})
target_compile_definitions(hydrogel_vbd_cpp_qt PRIVATE VBD_NO_OPENMP)
target_link_libraries(hydrogel_vbd_cpp_qt PRIVATE Eigen3::Eigen)
# 注意：不链接 OpenMP，不加 /openmp 标志
install(TARGETS hydrogel_vbd_cpp_qt DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src)
