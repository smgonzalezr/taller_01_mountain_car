"""
Entrenador reanudable del DQN.

Guarda tres cosas cada tramo: el agente, la curva de recompensas acumulada y el
buffer de repeticion. Al reanudar carga las tres, de modo que las trayectorias
exitosas ya vistas siguen disponibles para el replay aunque el proceso se haya
reiniciado. Esto permite entrenar en varias llamadas cortas sin perder progreso.

Uso repetido hasta llegar al total:

    uv run python scripts/train_dqn_resumable.py --total 2500 --run 900 --chunk 150

Cuando los episodios acumulados llegan a --total hace la evaluacion final.
"""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent

ENV_ID = "MountainCar-v0"
SAVE_DIR = Path("saves")
AGENT_PATH = SAVE_DIR / "dqn_mountaincar.pt"
REWARDS_PATH = SAVE_DIR / "dqn_rewards.npy"
BUFFER_PATH = SAVE_DIR / "dqn_buffer.pkl"
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
    parser.add_argument("--run", type=int, default=900, help="max episodes this invocation")
    parser.add_argument("--chunk", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    # Fresh start vs resume.
    if AGENT_PATH.exists():
        agent = DQNAgent.load(AGENT_PATH)
        all_rewards = list(np.load(REWARDS_PATH)) if REWARDS_PATH.exists() else []
        if BUFFER_PATH.exists():
            with open(BUFFER_PATH, "rb") as f:
                agent.buffer.buffer = pickle.load(f)
        print(
            f"Reanudando desde {agent.training_episodes} episodios | "
            f"buffer {len(agent.buffer)} | epsilon {agent.epsilon:.4f}",
            flush=True,
        )
    else:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        agent = DQNAgent(ENV_ID)
        all_rewards = []
        print("Inicio de entrenamiento nuevo", flush=True)

    start = len(all_rewards)
    target_this_run = min(args.total, start + args.run)
    t0 = time.time()

    while len(all_rewards) < target_this_run:
        n = min(args.chunk, target_this_run - len(all_rewards))
        rewards = agent.train(total_episodes=n, log_interval=n)
        all_rewards.extend(rewards)

        agent.save(AGENT_PATH)
        np.save(REWARDS_PATH, np.array(all_rewards, dtype=float))
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(agent.buffer.buffer, f)

        recent = float(np.mean(all_rewards[-100:]))
        print(
            f"[checkpoint] {len(all_rewards)}/{args.total} eps | "
            f"last100 avg {recent:.1f} | epsilon {agent.epsilon:.4f} | "
            f"elapsed {(time.time()-t0)/60:.1f} min",
            flush=True,
        )

    if len(all_rewards) >= args.total:
        print("Total alcanzado. Evaluando (modo codicioso) ...", flush=True)
        summary = evaluate(agent, args.eval_episodes)
        summary["agent"] = "dqn"
        summary["train_episodes"] = args.total
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            f"Pausa en {len(all_rewards)}/{args.total}. Vuelva a ejecutar para continuar.",
            flush=True,
        )


if __name__ == "__main__":
    main()
