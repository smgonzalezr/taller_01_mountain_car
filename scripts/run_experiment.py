"""
Entrena un agente, guarda el agente entrenado, la curva de recompensas y un
resumen de evaluacion. Sirve para reproducir de forma identica los resultados
que se reportan en el README.

Uso:

    uv run python scripts/run_experiment.py qlearning --episodes 20000
    uv run python scripts/run_experiment.py dqn --episodes 2500

Todo lo que produce queda en la carpeta saves/:

    saves/<agente>_mountaincar.<ext>   agente entrenado (formato del CLI)
    saves/<agente>_rewards.npy         recompensa por episodio de entrenamiento
    saves/<agente>_summary.json        resumen de la evaluacion final
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
SAVE_DIR = Path("saves")

AGENTS = {
    "qlearning": (QLearningAgent, SAVE_DIR / "qlearning_mountaincar.pkl"),
    "dqn": (DQNAgent, SAVE_DIR / "dqn_mountaincar.pt"),
}


def evaluate(agent, episodes: int = 100) -> dict:
    """Juega episodios en modo codicioso puro y resume el desempeno."""
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
    parser = argparse.ArgumentParser(description="Entrena y evalua un agente de MountainCar")
    parser.add_argument("agent", choices=tuple(AGENTS))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    import random

    random.seed(args.seed)

    cls, path = AGENTS[args.agent]
    agent = cls(ENV_ID)

    print(f"Entrenando {args.agent} por {args.episodes} episodios ...")
    t0 = time.time()
    rewards = agent.train(total_episodes=args.episodes)
    dt = time.time() - t0
    print(f"Entrenamiento terminado en {dt/60:.1f} min")

    agent.save(path)
    np.save(SAVE_DIR / f"{args.agent}_rewards.npy", np.array(rewards, dtype=float))

    print(f"Evaluando ({args.eval_episodes} episodios, modo codicioso) ...")
    summary = evaluate(agent, args.eval_episodes)
    summary["agent"] = args.agent
    summary["train_episodes"] = args.episodes
    summary["train_minutes"] = round(dt / 60, 2)

    with open(SAVE_DIR / f"{args.agent}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
