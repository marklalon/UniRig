import argparse
import os
import subprocess
import sys
from typing import List, Optional

DEFAULT_BPY_PYTHON = r"D:\AI\.bpy_env\Scripts\python.exe"


def get_bpy_python() -> str:
    return os.environ.get("UNIRIG_BPY_PYTHON", DEFAULT_BPY_PYTHON)


def run_module_in_bpy_env(module: str, module_args: List[str], cwd: Optional[str] = None, extra_env: Optional[dict] = None) -> int:
    bpy_python = get_bpy_python()
    if not os.path.exists(bpy_python):
        raise FileNotFoundError(
            f"bpy python not found: {bpy_python}. Set UNIRIG_BPY_PYTHON or install env at D:/AI/.bpy_env"
        )

    cmd = [bpy_python, "-m", module, *module_args]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    process = subprocess.run(cmd, cwd=cwd, env=env)
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Python module using bpy virtual environment")
    parser.add_argument("--module", required=True, help="Python module to run, e.g. src.data.extract")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to module, prefix with --")
    parsed = parser.parse_args()

    module_args = parsed.args
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]

    return run_module_in_bpy_env(parsed.module, module_args)


if __name__ == "__main__":
    sys.exit(main())
