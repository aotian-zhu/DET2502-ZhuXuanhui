"""Render a trained PyTorch DQN policy as an MP4 or PNG."""

import argparse
import json
import os

from agent import DeepQLearningAgent
from game_environment import Snake
from utils import visualize_game


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v17.1")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-time-limit", type=int, default=398)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(
        os.path.join("model_config", f"{args.version}.json"),
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    env = Snake(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=args.max_time_limit,
        obstacles=bool(config["obstacles"]),
        version=args.version,
    )
    agent = DeepQLearningAgent(
        board_size=config["board_size"],
        frames=config["frames"],
        n_actions=config["n_actions"],
        buffer_size=10,
        version=args.version,
        device=args.device,
    )
    agent.load_model(os.path.join("models", args.version), args.iteration)

    animate = not args.no_animation
    extension = "mp4" if animate else "png"
    output = args.output or os.path.join(
        "images", f"game_visual_{args.version}_{args.iteration}.{extension}"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    visualize_game(env, agent, path=output, animate=animate, fps=args.fps)
    print(output)


if __name__ == "__main__":
    main()
