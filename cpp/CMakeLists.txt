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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DNDEBUG /openmp")
endif()

# ── 拉取 Eigen（仅头文件） ──
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

# ── 拉取 pybind11 ──
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

# ── 头文件目录 ──
set(CPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── 源文件 ──
set(VBD_SOURCES
    src/elastic_energy.cpp
    src/physics_terms.cpp
    src/vbd_solver.cpp
    bindings/pybind_vbd.cpp
)

# ── pybind11 模块 ──
pybind11_add_module(vbd_solver_cpp ${VBD_SOURCES})

target_include_directories(vbd_solver_cpp PRIVATE ${CPP_INCLUDE_DIR})

target_link_libraries(vbd_solver_cpp PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# 安装到源目录（以便 Python 导入）
install(TARGETS vbd_solver_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src/hydrogel_vbd)
