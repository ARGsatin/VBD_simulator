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
set(PYBIND11_FINDPYTHON ON)  # 使用现代 FindPython（适配 Python 3.13+）
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

# ── 源文件 ──
set(VBD_SOURCES
    src/elastic_energy.cpp
    src/physics_terms.cpp
    src/vbd_solver.cpp
    bindings/pybind_vbd.cpp
)

# ── pybind11 模块 ──
pybind11_add_module(hydrogel_vbd_cpp ${VBD_SOURCES})

target_include_directories(hydrogel_vbd_cpp PRIVATE ${CPP_INCLUDE_DIR})

target_link_libraries(hydrogel_vbd_cpp PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# 安装到源目录（以便 Python 导入）
install(TARGETS hydrogel_vbd_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src)
