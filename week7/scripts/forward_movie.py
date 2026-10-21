"""Animate the saved forward Markov chain and print its t=1000 moments."""

import argparse
import json
from pathlib import Path

from scripts.diffusion import CENTRES

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def forward_frames(trajectory, levels):
    """Build frames with persistent sample colours and one fixed coordinate scale."""
    labels = np.argmin(np.sum((trajectory[0, :, None] - np.array(CENTRES)) ** 2, axis=-1), axis=1)
    colours = np.array(["#0072b2", "#d55e00", "#009e73", "#cc79a7"])[labels]
    limit = float(np.ceil(np.max(np.abs(trajectory))) + 1)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.13, top=0.88)
    points = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], c=colours, s=7, alpha=0.45, linewidths=0)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    title = ax.set_title("")
    fig.text(0.5, 0.04, "Same samples throughout · colours follow the initial modes", ha="center", fontsize=11)

    def update(frame):
        points.set_offsets(trajectory[frame])
        title.set_text(f"Forward diffusion · t = {levels[frame]}\n{trajectory.shape[1]:,} samples")
        return points, title

    update(0)
    return fig, update


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "artifacts" / "diffusion")
    parser.add_argument("--output", type=Path, default=root / "evidence" / "forward-diffusion.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    source = args.run / "forward.npz"
    with np.load(source) as saved:
        trajectory, levels = saved["trajectory"], saved["levels"]
    if levels[0] != 0 or levels[-1] != 1000 or np.any(np.diff(levels) <= 0):
        raise ValueError("saved levels must increase strictly from 0 to 1000")
    terminal = trajectory[-1].astype(np.float64)
    mean, variance = terminal.mean(axis=0), terminal.var(axis=0)
    for coordinate in range(2):
        print(f"t=1000 x{coordinate + 1} mean={mean[coordinate]:.6f} variance={variance[coordinate]:.6f}", flush=True)

    fig, update = forward_frames(trajectory, levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie = FuncAnimation(fig, update, frames=len(levels), interval=1000 / args.fps)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(args.output, writer=writer, dpi=100)
    metadata = {
        "source": str(source.resolve()), "output": str(args.output.resolve()),
        "samples": trajectory.shape[1], "levels": levels.tolist(), "fps": args.fps,
        "coordinate_limits": list(fig.axes[0].get_xlim()),
        "colours": "fixed nearest-target-centre labels assigned at t=0",
        "terminal_level": 1000, "terminal_mean": mean.tolist(), "terminal_variance": variance.tolist(),
        "variance_ddof": 0, "codec": "libx264", "crf": 18, "pixel_format": "yuv420p",
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
    }
    (args.run / "movie.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
