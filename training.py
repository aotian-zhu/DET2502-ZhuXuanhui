"""Train the PyTorch DQN agent on parallel Snake environments."""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v17.1")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--log-frequency", type=int, default=500)
    parser.add_argument("--training-games", type=int, default=128)
    parser.add_argument("--evaluation-games", type=int, default=8)
    parser.add_argument("--initial-games", type=int, default=512)
    parser.add_argument("--initial-frames-per-game", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume-iteration", type=int, default=None)
    parser.add_argument("--skip-initial-fill", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(version):
    config_path = os.path.join("model_config", f"{version}.json")
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(env, agent, n_actions, games):
    rewards, lengths, completed_games = play_game2(
        env,
        agent,
        n_actions,
        n_games=games,
        epsilon=-1,
        record=False,
        sample_actions=False,
        frame_mode=True,
        total_frames=-1,
        total_games=games,
    )
    return {
        "reward_mean": round(float(rewards) / completed_games, 2),
        "length_mean": round(float(lengths) / completed_games, 2),
        "games": completed_games,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.version)
    model_dir = os.path.join("models", args.version)
    log_path = os.path.join("model_logs", f"{args.version}.csv")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("model_logs", exist_ok=True)

    logs = {
        "iteration": [],
        "reward_mean": [],
        "length_mean": [],
        "games": [],
        "loss": [],
        "epsilon": [],
    }
    epsilon = args.epsilon_start
    start_iteration = args.resume_iteration or 0
    if args.resume_iteration is not None:
        if args.skip_initial_fill:
            raise ValueError("续训检查点未保存经验池，必须重新采集初始经验。")
        if os.path.exists(log_path):
            history = pd.read_csv(log_path)
            if not history.empty and int(history["iteration"].max()) != start_iteration:
                raise ValueError("续训迭代必须与现有日志的最后一次迭代一致，以免覆盖记录。")
            logs = history.to_dict(orient="list")
            if not history.empty:
                epsilon = max(
                    float(history.iloc[-1]["epsilon"]) * args.epsilon_decay,
                    args.epsilon_end,
                )

    agent = DeepQLearningAgent(
        board_size=config["board_size"],
        frames=config["frames"],
        n_actions=config["n_actions"],
        buffer_size=config["buffer_size"],
        version=args.version,
        device=args.device,
    )
    if args.resume_iteration is not None:
        agent.load_model(model_dir, args.resume_iteration)

    print(f"Training on {agent.device}; start_iteration={start_iteration}; epsilon={epsilon}")
    if agent.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(agent.device)}")

    if not args.skip_initial_fill:
        initial_env = SnakeNumpy(
            board_size=config["board_size"],
            frames=config["frames"],
            max_time_limit=config["max_time_limit"],
            games=args.initial_games,
            frame_mode=True,
            obstacles=bool(config["obstacles"]),
            version=args.version,
        )
        play_game2(
            initial_env,
            agent,
            config["n_actions"],
            n_games=args.initial_games,
            record=True,
            epsilon=epsilon,
            frame_mode=True,
            total_frames=args.initial_games * args.initial_frames_per_game,
        )

    training_env = SnakeNumpy(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        games=args.training_games,
        frame_mode=True,
        obstacles=bool(config["obstacles"]),
        version=args.version,
    )
    evaluation_env = SnakeNumpy(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        games=args.evaluation_games,
        frame_mode=True,
        obstacles=bool(config["obstacles"]),
        version=args.version,
    )

    for local_iteration in tqdm(range(1, args.episodes + 1)):
        iteration = start_iteration + local_iteration
        play_game2(
            training_env,
            agent,
            config["n_actions"],
            epsilon=epsilon,
            n_games=args.training_games,
            record=True,
            reward_type="current",
            frame_mode=True,
            total_frames=args.training_games,
            stateful=True,
        )
        loss = agent.train_agent(batch_size=args.batch_size, reward_clip=True)

        if iteration % args.log_frequency == 0 or local_iteration == args.episodes:
            metrics = evaluate(
                evaluation_env, agent, config["n_actions"], args.evaluation_games
            )
            logs["iteration"].append(iteration)
            logs["reward_mean"].append(metrics["reward_mean"])
            logs["length_mean"].append(metrics["length_mean"])
            logs["games"].append(metrics["games"])
            logs["loss"].append(loss)
            logs["epsilon"].append(epsilon)
            pd.DataFrame(logs).to_csv(log_path, index=False)
            agent.update_target_net()
            checkpoint = agent.save_model(model_dir, iteration)
            print(
                f"iteration={iteration} reward={metrics['reward_mean']} "
                f"length={metrics['length_mean']} loss={loss:.6f} "
                f"checkpoint={checkpoint}"
            )
            if agent.device.type == "cuda":
                print(
                    f"CUDA peak allocated MiB={torch.cuda.max_memory_allocated() / 2**20:.2f}; "
                    f"reserved MiB={torch.cuda.memory_reserved() / 2**20:.2f}"
                )
            epsilon = max(epsilon * args.epsilon_decay, args.epsilon_end)


if __name__ == "__main__":
    main()
