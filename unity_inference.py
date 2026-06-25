"""Run inference by launching a Unity executable with ML-Agents model overrides."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path

from inference_utils import create_unity_environment, suppress_native_output


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    command: list[str]
    returncode: int
    result_path: Path | None
    log_path: Path | None
    model_dir: Path | None
    score: tuple[int, int] | None
    loaded_models: dict[str, bool]
    log_text: str | None = None





def _prepare_model_dir(
    model_mapping: dict[str, str | Path],
    run_dir: Path,
) -> Path:
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for behavior_name, model_path in model_mapping.items():
        source = Path(model_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = model_dir / f"{behavior_name}.onnx"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    return model_dir


def discover_behavior_names(
    env_path: str | Path,
    headless: bool = False,
    timeout: int = 20,
    suppress_unity_output: bool = True,
) -> tuple[str, ...]:
    """Launch a Unity ML-Agents executable and return its behavior names."""
    try:
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Behavior-name discovery requires mlagents_envs. "
            "Run this from an ML-Agents Python environment."
        ) from exc

    env_executable = Path(env_path).expanduser().resolve()
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)
    output_context = (
        suppress_native_output()
        if suppress_unity_output
        else contextlib.nullcontext()
    )
    with output_context:
        env = create_unity_environment(
            env_executable,
            no_graphics=headless,
            channel=channel,
            show_unity_output=True,
            timeout_wait=timeout,
        )
        try:
            env.reset()
            return tuple(env.behavior_specs.keys())
        finally:
            env.close()


def print_behavior_names(
    env_path: str | Path,
    headless: bool = False,
    timeout: int = 20,
    suppress_unity_output: bool = True,
) -> tuple[str, ...]:
    behavior_names = discover_behavior_names(
        env_path,
        headless=headless,
        timeout=timeout,
        suppress_unity_output=suppress_unity_output,
    )
    print(f"\nDetected behavior names: {list(behavior_names)}")
    override_names = tuple(name.split("?", maxsplit=1)[0] for name in behavior_names)
    print(
        "Specify model input argument as dict, with keys matching behavior "
        f"names in {override_names}, and the values pointing to their ONNX files."
    )

    return behavior_names


def _parse_score(log: str) -> tuple[int, int] | None:
    patterns = [
        r"Score:\s*(\d+)\s*[-:]\s*(\d+)",
        r"blueScore\D+(\d+)\D+purpleScore\D+(\d+)",
        r"Blue\D+(\d+)\D+Purple\D+(\d+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, log, flags=re.IGNORECASE)
        if matches:
            left, right = matches[-1]
            return int(left), int(right)
    return None


def _loaded_models(log: str, behavior_names: tuple[str, ...]) -> dict[str, bool]:
    return {name: f"Overriding behavior {name}" in log for name in behavior_names}


def _create_video(input_files: str | Path, output_file: str | Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i", input_files,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]

    subprocess.run([str(part) for part in cmd], check=True)


def _parse_model_arg(model_arg: str) -> tuple[str, Path]:
    behavior_name, separator, model_path = model_arg.partition("=")
    if not separator or not behavior_name or not model_path:
        raise argparse.ArgumentTypeError(
            "Models must use BEHAVIOR_NAME=MODEL_PATH, for example "
            "teamLeft=models/left.onnx."
        )
    return behavior_name, Path(model_path)


def _read_log_text(log_path: Path, fallback: str, keep_log_file: bool) -> str:
    if log_path.exists():
        return log_path.read_text()
    if keep_log_file:
        log_path.write_text(fallback)
    return fallback


def simulate_match(
    models: dict[str, str | Path],
    env_path: str | Path,
    results_dir: str | Path,
    episode_limit: int = 9,
    timeout_limit: int = 60,  # seconds
    headless: bool = False,
    save_video: bool = False,
    keep_artifacts: bool = True,
    include_log_text: bool = False,
    print_behaviors_on_load_failure: bool = True,
) -> InferenceResult:
    """Run a single match in the Unity environment and return the result.

    Args:
        models: Mapping of behavior name → ONNX model path. 
            e.g. {"teamLeft": "models/Ann.onnx",
                  "teamRight": "models/Bob.onnx"}
            Each key must exactly match a behavior name defined in the Unity 
            build (e.g. "teamLeft" / "teamRight" for differentiated-team builds, 
            or "SoccerTwos" for single-policy builds).
             A mismatch causes the sim to quit immediately on launch. 
        env_path: Path to the Unity executable.
            e.g. "env/football-inference.app/Contents/MacOS/Chibi-Football" 
            on macOS.
        results_dir: Directory where per-run logs and model symlinks are saved.
        headless: Run without a display (-batchmode -nographics).
        episode_limit: Stop after this many episodes (0 = no limit).
            With many fields, the program quits once the field with the maximum
            number of episodes hits the episode limit.
        timeout_limit: Hard wall-clock timeout in seconds.
        keep_artifacts: Keep logs, model symlinks, and frames after the run.
            The Unity log is always read internally for score and model-load
            parsing, even when log artifacts are discarded.
        include_log_text: Include the full Unity log text in the result object.
        print_behaviors_on_load_failure: If any requested model does not load,
            print behavior names discovered from the executable. If ML-Agents
            discovery is unavailable, print a warning and continue.
    """
    root = Path.cwd()
    env_executable = Path(env_path).expanduser().resolve()
    if not env_executable.is_file():
        raise FileNotFoundError(env_executable)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_match_dir = Path(results_dir).expanduser().resolve() / timestamp
    result_match_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _prepare_model_dir(models, result_match_dir)
    log_path = result_match_dir / "unity.log"

    command = [str(env_executable)]
    if headless:
        command.extend(["-batchmode", "-nographics"])

    command.extend(
        [
            "-logFile", str(log_path),
            "--mlagents-override-model-directory", str(model_dir),
            "--mlagents-quit-on-load-failure",
            "--mlagents-quit-after-episodes", str(episode_limit),
            "--mlagents-quit-after-seconds", str(timeout_limit),
            "-screen-width", "1280",
            "-screen-height", "720",
        ]
    )
    if save_video:
        video_dir = result_match_dir / "frames"
        video_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--save-frames", str(video_dir)])

    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_limit + 30,
    )

    log_text = _read_log_text(
        log_path,
        fallback=completed.stdout,
        keep_log_file=keep_artifacts,
    )

    loaded_models = _loaded_models(log_text, tuple(models.keys()))
    if print_behaviors_on_load_failure and not all(loaded_models.values()):
        try:
            print_behavior_names(env_executable, headless=headless)
        except Exception as exc:
            print(f"Could not discover behavior names: {exc}")

    result = InferenceResult(
        command=command,
        returncode=completed.returncode,
        result_path=result_match_dir if keep_artifacts or save_video else None,
        log_path=log_path if keep_artifacts else None,
        model_dir=model_dir if keep_artifacts else None,
        score=_parse_score(log_text),
        loaded_models=loaded_models,
        log_text=log_text if include_log_text else None,
    )

    if save_video:
        input_frames = result_match_dir / "frames" / "frame_%06d.png"
        output_file = result_match_dir / f"{timestamp}-match.mp4"
        _create_video(input_frames, output_file)

    if not keep_artifacts:
        if not save_video:
            shutil.rmtree(result_match_dir)
        else:
            shutil.rmtree(result_match_dir / "frames")
            shutil.rmtree(result_match_dir / "models")
            log_path.unlink(missing_ok=True)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Path to the Unity executable")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=_parse_model_arg,
        metavar="BEHAVIOR=PATH",
        help="Behavior-name to ONNX path mapping. Repeat for each behavior.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where per-run outputs are written.",
    )
    parser.add_argument("--episode-limit", type=int, default=9)
    parser.add_argument("--timeout-limit", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--include-log-text",
        action="store_true",
        help="Include the full Unity log text in InferenceResult.log_text.",
    )
    parser.add_argument(
        "--discard-artifacts",
        action="store_true",
        help="Delete logs and model symlinks after collecting the result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simulate_match(
        models=dict(args.model),
        env_path=args.env,
        results_dir=args.results_dir,
        episode_limit=args.episode_limit,
        timeout_limit=args.timeout_limit,
        headless=args.headless,
        save_video=args.save_video,
        keep_artifacts=not args.discard_artifacts,
        include_log_text=args.include_log_text,
    )
    print(f"Return code: {result.returncode}")
    print(f"Score: {result.score}")
    print(f"Loaded models: {result.loaded_models}")
    print(f"Result path: {result.result_path}")


if __name__ == "__main__":
    main()
