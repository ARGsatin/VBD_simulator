# -*- coding: utf-8 -*-
"""C++ 求解器自动编译模块。

提供 CppBuilder 类，封装 cmake configure + build + deploy 全流程。
不依赖 Qt，可在 CLI 或 GUI 线程中调用。

使用方式
--------
.. code-block:: python

    from hydrogel_vbd.solver.cpp_builder import CppBuilder

    builder = CppBuilder()
    if not builder.pyd_exists():
        missing = builder.check_prerequisites()
        if missing:
            print(f"缺少: {missing}")
        else:
            result = builder.build(on_line=print)
            print("成功" if result.success else f"失败: {result.error_summary}")
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CppBuildResult:
    """一次编译的结果。"""
    success: bool
    pyd_path: Path | None = None
    log_lines: list[str] = field(default_factory=list)
    error_summary: str | None = None
    elapsed_seconds: float = 0.0


class CppBuilder:
    """C++ 求解器自动编译工具。

    封装了 cmake 配置、MSBuild 编译、产物部署的完整流程，
    并通过 ``on_line`` 回调提供实时输出。
    """

    # ── 路径常量 ──
    CPP_DIR = Path(__file__).resolve().parent.parent.parent.parent / "cpp"
    SRC_DIR = Path(__file__).resolve().parent.parent.parent  # src/

    def __init__(self) -> None:
        self._project_root = self.CPP_DIR.parent  # VBD_simulator/

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def pyd_exists(self) -> bool:
        """检查 src/ 下是否存在与当前 Python 版本匹配的 .pyd 文件。"""
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pattern = f"hydrogel_vbd_cpp.{tag}-*.pyd"
        matches = list(self.SRC_DIR.glob(pattern))
        return len(matches) > 0

    def check_prerequisites(self) -> list[str]:
        """检查编译所需的工具是否可用，返回缺失项的列表（空=全部就绪）。"""
        missing: list[str] = []

        if not shutil.which("cmake"):
            missing.append("cmake")

        # 检查 MSVC 或 clang-cl
        if not self._find_msvc_cl():
            missing.append("MSVC 编译器 (cl.exe)")

        # 检查 pybind11 pip 包（或至少可以安装）
        try:
            import pybind11  # noqa: F401
        except ImportError:
            missing.append("pybind11 (pip install pybind11)")

        return missing

    def build(
        self,
        on_line: "Callable[[str], None] | None" = None,
    ) -> CppBuildResult:
        """执行完整编译流程（同步，建议在后台线程中调用）。

        Parameters
        ----------
        on_line : callable or None
            每行输出回调，用于 GUI 实时显示。

        Returns
        -------
        CppBuildResult
        """
        import time
        t0 = time.perf_counter()
        result = CppBuildResult(success=False)

        def emit(line: str) -> None:
            result.log_lines.append(line)
            if on_line is not None:
                on_line(line)

        # ── 0. 获取 venv 信息 ──
        venv_root, venv_site = self._detect_venv()
        python_exe = (
            Path(venv_root) / "Scripts" / "python.exe"
            if venv_root
            else Path(sys.executable)
        )

        # ── 1. 确保 pybind11 已安装 ──
        emit("[build] 检查 pybind11 ...")
        try:
            import pybind11
            pybind11_dir = pybind11.get_cmake_dir()
            emit(f"[build]   [OK] pybind11 已安装 (cmake dir: {pybind11_dir})")
        except ImportError:
            emit("[build]   安装 pybind11 ...")
            proc = subprocess.run(
                [str(python_exe), "-m", "pip", "install", "pybind11"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(self._project_root),
                env=self._build_env(),
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-300:]
                result.error_summary = f"pybind11 安装失败: {stderr_tail}"
                emit(f"[build]   [FAIL] {result.error_summary}")
                result.elapsed_seconds = time.perf_counter() - t0
                return result
            import pybind11
            pybind11_dir = pybind11.get_cmake_dir()
            emit(f"[build]   [OK] pybind11 安装完成")

        # ── 2. 清理 + 创建 build 目录 ──
        build_dir = self.CPP_DIR / "build"
        emit(f"[build] 准备构建目录: {build_dir}")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        # ── 3. cmake configure ──
        emit("[build] cmake configure ...")
        cmake_args = [
            "cmake", "..",
            "-G", "Visual Studio 17 2022",
            "-A", "x64",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DPython_ROOT_DIR={venv_root}",
        ]
        emit(f"[build]   {' '.join(cmake_args)}")

        proc = subprocess.run(
            cmake_args,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(build_dir),
            env=self._build_env(),
        )
        for line in (proc.stdout or "").splitlines():
            stripped = line.strip()
            if stripped:
                emit(f"[build]   {stripped}")
        for line in (proc.stderr or "").splitlines():
            stripped = line.strip()
            if stripped:
                emit(f"[build]   [stderr] {stripped}")

        if proc.returncode != 0:
            result.error_summary = "cmake configure 失败，详见上方日志"
            emit(f"[build] [FAIL] {result.error_summary}")
            result.elapsed_seconds = time.perf_counter() - t0
            return result
        emit("[build]   [OK] cmake configure 完成")

        # ── 4. cmake build ──
        emit("[build] cmake build (Release) ...")
        proc = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(build_dir),
            env=self._build_env(),
        )
        for line in (proc.stdout or "").splitlines():
            stripped = line.strip()
            if stripped:
                emit(f"[build]   {stripped}")
        for line in (proc.stderr or "").splitlines():
            stripped = line.strip()
            if stripped:
                emit(f"[build]   [stderr] {stripped}")

        if proc.returncode != 0:
            # 提取 MSVC 错误行
            stdout_text = proc.stdout or ""
            error_lines = [l for l in stdout_text.splitlines()
                          if "error C" in l or "error MSB" in l or "error LNK" in l]
            if error_lines:
                result.error_summary = error_lines[-1].strip()
            else:
                result.error_summary = "cmake build 失败，详见上方日志"
            emit(f"[build] [FAIL] {result.error_summary}")
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # ── 5. 部署 .pyd 到 src/ ──
        emit("[build] 部署 .pyd -> src/ ...")
        pyd_files = list(build_dir.glob("Release/*.pyd"))
        if not pyd_files:
            result.error_summary = "编译完成但未生成 .pyd 文件"
            emit(f"[build] [FAIL] {result.error_summary}")
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        pyd_path = pyd_files[0]
        dest = self.SRC_DIR / pyd_path.name
        # 删除旧版本（如有）
        for old in self.SRC_DIR.glob("hydrogel_vbd_cpp.*.pyd"):
            try:
                old.unlink()
            except OSError:
                pass
        shutil.copy2(pyd_path, dest)
        emit(f"[build]   {pyd_path.name} -> {dest}")

        result.success = True
        result.pyd_path = dest
        result.elapsed_seconds = time.perf_counter() - t0
        emit(f"[build] [OK] 编译成功 ({result.elapsed_seconds:.1f}s)")
        return result

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _find_msvc_cl() -> str | None:
        """在 PATH 中查找 MSVC 的 cl.exe。"""
        cl = shutil.which("cl.exe") or shutil.which("cl")
        if cl:
            return cl
        # 尝试常见的 VS 安装路径
        candidates = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC"),
        ]
        for base in candidates:
            if base.exists():
                # 查找最新版本的 cl.exe
                versions = sorted(base.glob("*"), reverse=True)
                for v in versions:
                    cl_path = v / "bin" / "Hostx64" / "x64" / "cl.exe"
                    if cl_path.exists():
                        return str(cl_path)
        return None

    @staticmethod
    def _detect_venv() -> tuple[str | None, str | None]:
        """检测当前是否在 venv 中，返回 (venv_root, site_packages)。"""
        prefix = sys.prefix
        base_prefix = getattr(sys, "base_prefix", prefix)
        if prefix == base_prefix:
            # 不在 venv 中
            return None, None
        # 在 venv 中
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else ""
        return prefix, site_packages

    @staticmethod
    def _build_env() -> dict[str, str]:
        """构造编译用的环境变量（继承当前环境，确保 MSVC 在 PATH 中）。"""
        env = os.environ.copy()
        # 移除可能导致混淆的 MSYS2/MINGW 路径
        path_parts = env.get("PATH", "").split(os.pathsep)
        filtered = [
            p for p in path_parts
            if "msys2" not in p.lower() and "mingw" not in p.lower()
        ]
        env["PATH"] = os.pathsep.join(filtered)
        # 确保 VCToolsVersion 等 VS 环境变量不被覆盖
        return env
