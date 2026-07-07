"""Plotting utilities for simulation logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_summary_plots(log: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    time = log["time"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(time, log["u_cmd"], label="target u")
    axes[0].plot(time, log["u_actual"], label="actual u", linewidth=1.2)
    axes[0].set_ylabel("board u [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, 1000.0 * (log["u_actual"] - log["u_cmd"]))
    axes[1].set_ylabel("tracking error [mm]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, log["normal_force"])
    force_band = log.get("force_band")
    if force_band is not None:
        lower, upper = np.asarray(force_band)[0]
    else:
        lower, upper = 0.08, 0.25
    axes[2].axhspan(lower, upper, color="tab:green", alpha=0.15, label="desired band")
    axes[2].set_ylabel("normal force [N]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time, 1000.0 * log["surface_gap"])
    axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[3].set_ylabel("contact gap [mm]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "tracking_force.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time, np.rad2deg(log["board_angle"]))
    axes[0].set_ylabel("board angle [deg]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, log["q_cmd"][:, 0], label="q1 cmd")
    axes[1].plot(time, log["q_cmd"][:, 1], label="q2 cmd")
    axes[1].plot(time, log["q_cmd"][:, 2], label="q3 cmd")
    axes[1].set_ylabel("joint targets [rad]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="best", ncol=3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "board_and_joints.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(log["target_xz"][:, 0], log["target_xz"][:, 1], label="world target")
    ax.plot(log["pen_xz"][:, 0], log["pen_xz"][:, 1], label="pen tip", linewidth=1.1)
    if "contact_xz" in log:
        ax.plot(log["contact_xz"][:, 0], log["contact_xz"][:, 1], label="contact point", linewidth=1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "world_path.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate figures from a saved simulation npz log.")
    parser.add_argument("log", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "figures" / "simulation_log.npz")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    data = np.load(args.log)
    log = {key: data[key] for key in data.files}
    for path in save_summary_plots(log, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
