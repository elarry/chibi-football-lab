"""
Run inference through the ML-Agents Python API with two ONNXRuntime policies.

Usage (macOS — drop the .app extension):
    python mlagents_inference.py \
        --env ./Builds/football-inference \
        --left models/teamLeft.onnx \
        --right models/teamRight.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from inference_utils import create_unity_environment, suppress_native_output


@dataclass
class OnnxPolicy:
    session: ort.InferenceSession
    input_names: set[str]
    output_names: list[str]
    mask_size: int | None

    @classmethod
    def load(cls, path: str | Path) -> OnnxPolicy:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_names = {i.name for i in session.get_inputs()}
        output_names = [o.name for o in session.get_outputs()]
        mask_size = None
        if "action_masks" in input_names:
            mask_size = next(
                i for i in session.get_inputs() if i.name == "action_masks"
            ).shape[1]
        return cls(session, input_names, output_names, mask_size)

    def act(self, decision_steps: DecisionSteps) -> ActionTuple:
        n_agents = len(decision_steps)
        feed: dict[str, np.ndarray] = {}

        for idx, obs_batch in enumerate(decision_steps.obs):
            name = f"obs_{idx}"
            if name in self.input_names:
                feed[name] = obs_batch.astype(np.float32)

        if self.mask_size is not None:
            feed["action_masks"] = np.ones(
                (n_agents, self.mask_size),
                dtype=np.float32,
            )

        results = self.session.run(self.output_names, feed)
        output_map = dict(zip(self.output_names, results, strict=True))

        continuous = None
        discrete = None
        for name, value in output_map.items():
            if name == "deterministic_discrete_actions":
                discrete = value.astype(np.int32)
            elif name == "discrete_actions" and discrete is None:
                discrete = value.astype(np.int32)
            elif name == "continuous_actions":
                continuous = value

        return ActionTuple(continuous=continuous, discrete=discrete)


class Team(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def label(self) -> str:
        return "Left " if self is Team.LEFT else "Right"


@dataclass(frozen=True)
class BehaviorRunner:
    behavior_name: str
    team: Team
    policy: OnnxPolicy


@dataclass(frozen=True)
class TerminalEvent:
    team: Team
    agent_id: int
    reward: float
    group_reward: float
    terminal_group_reward: float
    outcome: str
    left_score: int
    right_score: int

    def format(self) -> str:
        return (
            f"  [{self.team.label}] agent {self.agent_id}: "
            f"reward={self.reward:+.2f} "
            f"group_reward={self.group_reward:+.2f} "
            f"terminal_group={self.terminal_group_reward:+.2f} "
            f"({self.outcome})  "
            f"score: Left {self.left_score} - {self.right_score} Right"
        )


class ScoreTracker:
    def __init__(self) -> None:
        self.episode = 0
        self.goals = {Team.LEFT: 0, Team.RIGHT: 0}
        self.cumulative_rewards: dict[tuple[str, int], float] = {}
        self.cumulative_group_rewards: dict[tuple[str, int], float] = {}
        self.scored_this_episode: set[Team] = set()

    def record_decisions(
        self,
        behavior_name: str,
        decision_steps: DecisionSteps,
    ) -> None:
        for agent_id, reward, group_reward in zip(
            decision_steps.agent_id,
            decision_steps.reward,
            get_group_rewards(decision_steps),
            strict=True,
        ):
            self._add_reward(behavior_name, agent_id, reward, group_reward)

    def record_terminals(
        self,
        behavior_name: str,
        terminal_steps: TerminalSteps,
        team: Team,
    ) -> list[TerminalEvent]:
        events: list[TerminalEvent] = []
        for agent_id, reward, group_reward, interrupted in zip(
            terminal_steps.agent_id,
            terminal_steps.reward,
            get_group_rewards(terminal_steps),
            terminal_steps.interrupted,
            strict=True,
        ):
            key = self._add_reward(behavior_name, agent_id, reward, group_reward)
            total_reward = self.cumulative_rewards.pop(key)
            total_group_reward = self.cumulative_group_rewards.pop(key)
            outcome = self._classify_outcome(team, group_reward, interrupted)

            events.append(
                TerminalEvent(
                    team=team,
                    agent_id=int(agent_id),
                    reward=total_reward,
                    group_reward=total_group_reward,
                    terminal_group_reward=float(group_reward),
                    outcome=outcome,
                    left_score=self.goals[Team.LEFT],
                    right_score=self.goals[Team.RIGHT],
                )
            )
        return events

    def finish_episode(self) -> str:
        self.episode += 1
        self.scored_this_episode.clear()
        return (
            f"Episode {self.episode} ended. "
            f"Score: Left {self.goals[Team.LEFT]} - "
            f"{self.goals[Team.RIGHT]} Right"
        )

    def final_score(self) -> str:
        return f"Left Team {self.goals[Team.LEFT]} - {self.goals[Team.RIGHT]} Right Team"

    def _add_reward(
        self,
        behavior_name: str,
        agent_id: int,
        reward: float,
        group_reward: float,
    ) -> tuple[str, int]:
        key = (behavior_name, int(agent_id))
        self.cumulative_rewards[key] = (
            self.cumulative_rewards.get(key, 0.0) + float(reward)
        )
        self.cumulative_group_rewards[key] = (
            self.cumulative_group_rewards.get(key, 0.0) + float(group_reward)
        )
        return key

    def _classify_outcome(
        self,
        team: Team,
        group_reward: float,
        interrupted: bool,
    ) -> str:
        if group_reward > 0 and team not in self.scored_this_episode:
            self.goals[team] += 1
            self.scored_this_episode.add(team)
            return "GOAL"
        if group_reward > 0:
            return "GOAL"
        if group_reward < 0:
            return "conceded"
        if interrupted:
            return "timeout"
        return "end"


def find_behavior(behavior_names: list[str], keyword: str) -> str:
    match = [b for b in behavior_names if keyword.lower() in b.lower()]
    if not match:
        raise ValueError(
            f"No behavior containing '{keyword}' found. "
            f"Available: {behavior_names}"
        )
    return match[0]


def build_runners(
    behavior_names: list[str],
    left_policy: OnnxPolicy,
    right_policy: OnnxPolicy,
) -> list[BehaviorRunner]:
    left_behavior = find_behavior(behavior_names, Team.LEFT.value)
    right_behavior = find_behavior(behavior_names, Team.RIGHT.value)

    return [
        BehaviorRunner(left_behavior, Team.LEFT, left_policy),
        BehaviorRunner(right_behavior, Team.RIGHT, right_policy),
    ]


def get_group_rewards(steps: DecisionSteps | TerminalSteps) -> np.ndarray:
    group_rewards = getattr(steps, "group_reward", None)
    if group_rewards is None:
        return np.zeros(len(steps), dtype=np.float32)
    return group_rewards


def simulate_match(
    env_path: str | Path,
    left_model_path: str | Path,
    right_model_path: str | Path,
    no_graphics: bool,
    time_scale: float,
    verbose: bool,
    show_unity_output: bool = False,
) -> None:
    left_policy = OnnxPolicy.load(left_model_path)
    right_policy = OnnxPolicy.load(right_model_path)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=time_scale)

    env = create_unity_environment(env_path, no_graphics, channel, show_unity_output)
    if show_unity_output:
        env.reset()
    else:
        with suppress_native_output():
            env.reset()

    behavior_names = list(env.behavior_specs.keys())
    print(f"Detected behaviors: {behavior_names}")

    runners = build_runners(behavior_names, left_policy, right_policy)
    for runner in runners:
        print(f"{runner.team.label}→ {runner.behavior_name}")

    tracker = ScoreTracker()

    try:
        while True:
            episode_ended = False

            for runner in runners:
                decision_steps, terminal_steps = env.get_steps(runner.behavior_name)
                tracker.record_decisions(runner.behavior_name, decision_steps)

                events = tracker.record_terminals(
                    runner.behavior_name,
                    terminal_steps,
                    runner.team,
                )
                if verbose:
                    for event in events:
                        print(event.format())

                if len(terminal_steps) > 0:
                    episode_ended = True

                if len(decision_steps) > 0:
                    action_tuple = runner.policy.act(decision_steps)
                    env.set_actions(runner.behavior_name, action_tuple)

            if episode_ended:
                print(f"{tracker.finish_episode()}")

            env.step()

    except KeyboardInterrupt:
        print(f"\nStopped. Final score: {tracker.final_score()}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        required=True,
        help="Path to Unity executable (no .app on macOS)",
    )
    parser.add_argument("--left", required=True, help="Path to teamLeft ONNX model")
    parser.add_argument("--right", required=True, help="Path to teamRight ONNX model")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Simulation speed multiplier (default 1.0)",
    )
    parser.add_argument("--no-graphics", action="store_true", help="Run headless")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional reward and per-player stats.",
    )
    parser.add_argument(
        "--show-unity-output",
        action="store_true",
        help="Show stdout/stderr from the Unity executable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulate_match(
        args.env,
        args.left,
        args.right,
        args.no_graphics,
        args.time_scale,
        args.verbose,
        args.show_unity_output,
    )


if __name__ == "__main__":
    main()
