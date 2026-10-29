import subprocess
import sys
from pathlib import Path
import torch

def restart_in_project_venv_for_cuda(requested: str | None, module: str) -> None:
    if requested not in (None, 'cuda') or torch.cuda.is_available():
        return
    project_root = Path(__file__).resolve().parents[2]
    venv_python = project_root / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.is_file():
        return
    try:
        already_in_project_venv = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        already_in_project_venv = False
    if already_in_project_venv:
        return
    command = [str(venv_python), '-m', module, *sys.argv[1:]]
    print(f"CUDA is unavailable in the current Python environment:\n  {sys.executable}\nRestarting with the project's CUDA-enabled environment:\n  {venv_python}", flush=True)
    completed = subprocess.run(command, cwd=project_root)
    raise SystemExit(completed.returncode)

def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            project_root = Path(__file__).resolve().parents[2]
            venv_python = project_root / '.venv' / 'Scripts' / 'python.exe'
            raise RuntimeError(f'CUDA was requested, but the active Python uses a CPU-only PyTorch build. Active Python: {sys.executable}. Run with `{venv_python} -m scripts.run_experiment`, or install a CUDA-enabled PyTorch build in the active environment.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device {index} was requested, but only {torch.cuda.device_count()} CUDA device(s) are available.')
    return device
