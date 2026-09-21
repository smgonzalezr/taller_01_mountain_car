"""
Genera las figuras de evidencia a partir de las curvas de recompensa guardadas.

Produce en la carpeta docs/figures/:

    qlearning_curva.png    recompensa por episodio del Q-Learning tabular
    dqn_curva.png          recompensa por episodio del DQN
    comparacion.png        promedio movil de ambos metodos en un solo grafico

Uso:

    uv run --with matplotlib python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = Path("saves")
OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVED = -110


def moving_average(x: np.ndarray, w: int = 100) -> np.ndarray:
    if len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="valid")


def single_curve(name: str, color: str, title: str, out: str) -> None:
    rewards = np.load(SAVE_DIR / f"{name}_rewards.npy")
    ma = moving_average(rewards, 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, color=color, alpha=0.20, linewidth=0.8, label="recompensa por episodio")
    ax.plot(
        np.arange(len(ma)) + 99,
        ma,
        color=color,
        linewidth=2.2,
        label="promedio movil (100 episodios)",
    )
    ax.axhline(SOLVED, color="green", linestyle="--", linewidth=1.2, label="umbral resuelto (-110)")
    ax.axhline(-200, color="gray", linestyle=":", linewidth=1.0, label="peor caso (-200)")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Recompensa total del episodio")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out, dpi=130)
    plt.close(fig)
    print("escrito", OUT_DIR / out)


def comparison() -> None:
    ql = np.load(SAVE_DIR / "qlearning_rewards.npy")
    dqn = np.load(SAVE_DIR / "dqn_rewards.npy")
    ql_ma = moving_average(ql, 100)
    dqn_ma = moving_average(dqn, 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(ql_ma)) + 99, ql_ma, color="#1f77b4", linewidth=2.2, label="Q-Learning tabular")
    ax.plot(np.arange(len(dqn_ma)) + 99, dqn_ma, color="#d62728", linewidth=2.2, label="DQN")
    ax.axhline(SOLVED, color="green", linestyle="--", linewidth=1.2, label="umbral resuelto (-110)")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Promedio movil de la recompensa (100 episodios)")
    ax.set_title("Q-Learning tabular frente a DQN en MountainCar-v0")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparacion.png", dpi=130)
    plt.close(fig)
    print("escrito", OUT_DIR / "comparacion.png")


if __name__ == "__main__":
    single_curve("qlearning", "#1f77b4", "Q-Learning tabular en MountainCar-v0", "qlearning_curva.png")
    single_curve("dqn", "#d62728", "DQN en MountainCar-v0", "dqn_curva.png")
    comparison()
