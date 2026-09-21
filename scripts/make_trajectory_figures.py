"""
Evidencia del mejor resultado: grafica la posicion del carro paso a paso en el
mejor episodio codicioso de cada agente. La linea de la meta esta en 0.5, asi se
ve el vaiven que toma impulso y como el carro llega arriba.

Uso:

    uv run --with matplotlib python scripts/make_trajectory_figures.py
"""
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mountain_car.agents import DQNAgent, QLearningAgent

OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = {
    "qlearning": (QLearningAgent, Path("saves/qlearning_mountaincar.pkl"), "#1f77b4"),
    "dqn": (DQNAgent, Path("saves/dqn_mountaincar.pt"), "#d62728"),
}


def best_trajectory(agent, tries: int = 40):
    env = gym.make("MountainCar-v0")
    best = None
    for _ in range(tries):
        obs, _ = env.reset()
        positions = [float(obs[0])]
        total, done = 0.0, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            positions.append(float(obs[0]))
            total += reward
            done = terminated or truncated
        if best is None or total > best[0]:
            best = (total, positions)
    env.close()
    return best


for name, (cls, path, color) in AGENTS.items():
    agent = cls.load(path)
    total, positions = best_trajectory(agent)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(positions, color=color, linewidth=2)
    ax.axhline(0.5, color="green", linestyle="--", linewidth=1.3, label="meta (posicion 0.5)")
    ax.set_xlabel("Paso dentro del episodio")
    ax.set_ylabel("Posicion del carro")
    ax.set_title(f"{name}: mejor episodio, recompensa {total:.0f}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"{name}_trayectoria.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("escrito", out, "| recompensa", total)
