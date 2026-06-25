from __future__ import annotations

import contextlib
import os
import sys
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def clear_quarantine(path: str | Path) -> None:
    """Remove macOS quarantine attrs from Unity app bundles, if xattr exists."""
    xattr = shutil.which("xattr")
    if not xattr:
        return
    subprocess.run([xattr, "-cr", str(path)], check=False)


def mlagents_env_path(env_executable: str | Path) -> str:
    """Return the path format expected by UnityEnvironment on macOS app bundles."""
    path = Path(env_executable).expanduser().resolve()
    for parent in (path, *path.parents):
        if parent.suffix == ".app":
            return str(parent.with_suffix(""))
    return str(path)


@contextlib.contextmanager
def suppress_native_output() -> Iterator[None]:
    """Suppress output written directly to process stdout/stderr file descriptors."""
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def create_unity_environment(
    env_path: str | Path,
    no_graphics: bool,
    channel: Any,
    show_unity_output: bool = False,
    timeout_wait: int | None = None,
) -> Any:
    """Create a UnityEnvironment with optional stdout/stderr suppression."""
    from mlagents_envs.environment import UnityEnvironment

    kwargs: dict[str, Any] = {
        "file_name": mlagents_env_path(env_path),
        "no_graphics": no_graphics,
        "side_channels": [channel],
        "seed": 0,
    }
    if timeout_wait is not None:
        kwargs["timeout_wait"] = timeout_wait

    if show_unity_output:
        return UnityEnvironment(**kwargs)

    with suppress_native_output():
        return UnityEnvironment(**kwargs)
