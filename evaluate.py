"""Evaluate a saved PyTorch DQN checkpoint on parallel Snake games."""

import argparse
import json
import os

import numpy as np

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v17.1")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    with open(
        os.path.join("model_config", f"{args.version}.json"),
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    agent = DeepQLearningAgent(
        board_size=config["board_size"],
        frames=config["frames"],
        n_actions=config["n_actions"],
        buffer_size=10,
        version=args.version,
        device=args.device,
    )
    agent.load_model(os.path.join("models", args.version), args.iteration)
    env = SnakeNumpy(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        games=args.games,
        frame_mode=True,
        obstacles=bool(config["obstacles"]),
        version=args.version,
    )
    rewards, lengths, completed_games = play_game2(
        env,
        agent,
        config["n_actions"],
        n_games=args.games,
        epsilon=-1,
        record=False,
        frame_mode=True,
        total_frames=-1,
        total_games=args.games,
    )
    print(f"games={completed_games}")
    print(f"mean_reward={float(rewards) / completed_games:.4f}")
    print(f"mean_length={float(lengths) / completed_games:.4f}")


if __name__ == "__main__":
    main()
