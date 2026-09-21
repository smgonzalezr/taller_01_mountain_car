"""
Entrenador del DQN por tramos, dentro de un solo proceso, con checkpoints.

Mantiene un unico objeto agente (y por tanto un unico buffer de repeticion)
durante toda la corrida, y cada cierto numero de episodios guarda el agente y
la curva de recompensas acumulada. Asi, si la corrida se interrumpe, en disco
queda el ultimo checkpoint utilizable junto con su curva.

Uso:

    uv run python scripts/train_dqn_checkpointed.py --total 2500 --chunk 250
"""
import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents import DQNAgent

ENV_ID = "MountainCar-v0"
SAVE_DIR = Path("saves")
AGENT_PATH = SAVE_DIR / "dqn_mountaincar.pt"
REWARDS_PATH = SAVE_DIR / "dqn_rewards.npy"
SUMMARY_PATH = SAVE_DIR / "dqn_summary.json"


def evaluate(agent, episodes: int = 100) -> dict:
    env = gym.make(ENV_ID)
    rewards, reached = [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        reached += int(terminated)
    env.close()
    rewards = np.array(rewards, dtype=float)
    return {
        "episodes": episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "best_reward": float(rewards.max()),
        "worst_reward": float(rewards.min()),
        "reached_flag": int(reached),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2500)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)

    agent = DQNAgent(ENV_ID)
    all_rewards: list[float] = []

    t0 = time.time()
    done_eps = 0
    while done_eps < args.total:
        n = min(args.chunk, args.total - done_eps)
        rewards = agent.train(total_episodes=n, log_interval=n)
        all_rewards.extend(rewards)
        done_eps += n

        agent.save(AGENT_PATH)
        np.save(REWARDS_PATH, np.array(all_rewards, dtype=float))
        recent = np.mean(all_rewards[-100:])
        print(
            f"[checkpoint] {done_eps}/{args.total} eps | "
            f"last100 avg {recent:.1f} | epsilon {agent.epsilon:.4f} | "
            f"elapsed {(time.time()-t0)/60:.1f} min",
            flush=True,
        )

    print("Evaluando (modo codicioso) ...", flush=True)
    summary = evaluate(agent, args.eval_episodes)
    summary["agent"] = "dqn"
    summary["train_episodes"] = args.total
    summary["train_minutes"] = round((time.time() - t0) / 60, 2)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
