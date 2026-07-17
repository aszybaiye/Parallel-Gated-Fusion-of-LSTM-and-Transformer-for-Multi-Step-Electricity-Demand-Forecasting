import os
import shutil


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_output_root() -> str:
    return os.path.join(get_project_root(), "Output")


def output_path(*parts: str) -> str:
    return os.path.join(get_output_root(), *parts)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def reset_output_root() -> str:
    out = get_output_root()
    if os.path.exists(out):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    return out
