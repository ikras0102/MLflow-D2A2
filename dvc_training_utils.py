import os
import shutil
import subprocess


def _get_arg(args, name, default=None):
    return getattr(args, name, default)


def data_paths_from_args(args):
    paths = []
    dataset_dir = _get_arg(args, "dataset_dir", None)
    if dataset_dir:
        paths.append(dataset_dir)

    extra_paths = _get_arg(args, "dvc_data_paths", None)
    if extra_paths:
        for path in extra_paths.split(","):
            path = path.strip()
            if path:
                paths.append(path)

    deduped = []
    seen = set()
    for path in paths:
        normalized = os.path.normpath(path)
        if normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _run(command, cwd=None):
    print("running:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError("command failed: %s" % " ".join(command))
    return completed


def _repo_root():
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return os.getcwd()


def prepare_dvc_data(args):
    repo_root = _repo_root()
    data_paths = data_paths_from_args(args)
    dvc_bin = shutil.which("dvc")
    dvc_pull = bool(_get_arg(args, "dvc_pull", False))
    dvc_required = bool(_get_arg(args, "dvc_required", False))

    if (dvc_pull or dvc_required) and dvc_bin is None:
        raise RuntimeError(
            "DVC is required for this run, but `dvc` was not found. "
            "Install the pinned DVC version from environment.yaml first."
        )

    if dvc_pull:
        command = ["dvc", "pull"]
        dvc_remote = _get_arg(args, "dvc_remote", None)
        if dvc_remote:
            command.extend(["-r", dvc_remote])
        dvc_jobs = _get_arg(args, "dvc_jobs", None)
        if dvc_jobs:
            command.extend(["-j", str(dvc_jobs)])
        command.extend(data_paths)
        _run(command, cwd=repo_root)

    missing_paths = [
        path for path in data_paths if not os.path.exists(os.path.join(repo_root, path))
    ]
    if missing_paths:
        hint = "Missing data paths: %s" % ", ".join(missing_paths)
        if dvc_bin is not None:
            hint += ". Try running with --dvc_pull."
        else:
            hint += ". Install DVC and run `dvc pull` first."
        raise RuntimeError(hint)

    if dvc_required and dvc_bin is not None:
        _run(["dvc", "status"], cwd=repo_root)

    return data_paths
