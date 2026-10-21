"""Generate DDPM samples, animate their reverse process, and measure the modes."""

import argparse
import csv
import json
from pathlib import Path
import time

from scripts.diffusion import CENTRES, COMPONENT_VARIANCE, STEPS, sample_mixture
from scripts.noise_model import ddpm_trajectory

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


COLOURS = np.array(["#0072b2", "#d55e00", "#009e73", "#cc79a7"])


def mode_statistics(samples):
    """Assign points to nearest target centres; spreads use population moments."""
    samples = np.asarray(samples, dtype=np.float64)
    labels = np.argmin(np.sum((samples[:, None] - np.array(CENTRES)) ** 2, axis=-1), axis=1)
    rows = []
    for mode, centre in enumerate(np.array(CENTRES)):
        group = samples[labels == mode]
        mean, variance, std = group.mean(0), group.var(0), group.std(0)
        rows.append({"mode": mode + 1, "target_x1": float(centre[0]), "target_x2": float(centre[1]),
                     "count": len(group), "fraction": len(group) / len(samples),
                     "mean_x1": float(mean[0]), "mean_x2": float(mean[1]),
                     "std_x1": float(std[0]), "std_x2": float(std[1]),
                     "variance_x1": float(variance[0]), "variance_x2": float(variance[1])})
    return labels, rows


def reverse_frames(trajectory):
    """Follow the same points on fixed axes, colored by their final nearest mode."""
    labels = np.argmin(np.sum((trajectory[-1, :, None] - np.array(CENTRES)) ** 2, axis=-1), axis=1)
    limit = float(np.ceil(max(np.max(np.abs(trajectory)), 3.)) + 1)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.13, top=0.88)
    points = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], c=COLOURS[labels], s=7, alpha=0.45, linewidths=0)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    title = ax.set_title("")
    fig.text(0.5, 0.04, "Same samples throughout · colours identify final nearest target modes", ha="center", fontsize=10)

    def update(frame):
        points.set_offsets(trajectory[frame])
        title.set_text(f"DDPM reverse diffusion · t = {STEPS - frame}\n{trajectory.shape[1]:,} samples · final trained predictor")
        return points, title

    update(0)
    return fig, update


def plot_comparison(target, generated, labels, rows, output):
    centres = np.array(CENTRES)
    target_labels = np.argmin(np.sum((target[:, None] - centres) ** 2, axis=-1), axis=1)
    limit = float(np.ceil(max(np.max(np.abs(target)), np.max(np.abs(generated)), 3.)) + 1)
    fig = plt.figure(figsize=(10, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    for index, (samples, groups, title) in enumerate(zip((target, generated), (target_labels, labels),
                                                        ("Independent target samples", "Trained DDPM samples"))):
        ax = fig.add_subplot(grid[0, index])
        ax.scatter(samples[:, 0], samples[:, 1], c=COLOURS[groups], s=5, alpha=0.35, linewidths=0)
        ax.scatter(centres[:, 0], centres[:, 1], c="black", marker="+", s=80, linewidths=1.5, label="target centres")
        if index == 1:
            ax.scatter([r["mean_x1"] for r in rows], [r["mean_x2"] for r in rows],
                       marker="x", c="black", s=45, linewidths=1.2, label="measured centres")
        ax.set(title=title, xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
        ax.legend(fontsize=8, loc="upper right")
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    cells = [[f"({r['target_x1']:.0f}, {r['target_x2']:.0f})", f"({r['mean_x1']:.3f}, {r['mean_x2']:.3f})",
              f"{r['count']} ({100*r['fraction']:.1f}%)", f"{r['std_x1']:.3f}", f"{r['std_x2']:.3f}"] for r in rows]
    table = table_ax.table(cellText=cells, colLabels=["Target centre", "Measured centre", "Population", "Std x1", "Std x2"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table_ax.set_title(f"Target per mode: {len(generated)/4:g} points in expectation (25%); coordinate std = 0.3", fontsize=10)
    fig.suptitle(f"Final DDPM distribution · {len(generated):,} samples · 1000 reverse steps")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "noise-model" / "checkpoint-025000.npz")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "reverse-diffusion")
    parser.add_argument("--figure", type=Path, default=root / "evidence" / "reverse-distribution.png")
    parser.add_argument("--movie", type=Path, default=root / "evidence" / "reverse-diffusion.mp4")
    args = parser.parse_args()
    if args.samples < 1 or args.frame_stride < 1:
        parser.error("--samples and --frame-stride must be positive")
    with np.load(args.checkpoint) as saved:
        params = {name: jnp.array(saved[name]) for name in saved.files}
    args.output.mkdir(parents=True, exist_ok=False)
    sampling_key, target_key = jax.random.split(jax.random.key(args.seed))
    started = time.perf_counter()
    reverse_key, initial_key = jax.random.split(sampling_key)
    initial = jax.random.normal(initial_key, (args.samples, 2), dtype=jnp.float32)
    trajectory = np.array(ddpm_trajectory(params, initial, reverse_key))
    compute_seconds = time.perf_counter() - started
    if not np.isfinite(trajectory).all():
        raise ValueError("trained predictor produced non-finite DDPM samples")
    target = np.array(sample_mixture(target_key, args.samples))
    labels, rows = mode_statistics(trajectory[-1])
    np.savez_compressed(args.output / "trajectory.npz", trajectory=trajectory, levels=np.arange(STEPS, -1, -1), target=target)
    with (args.output / "mode-statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"target=({row['target_x1']:g},{row['target_x2']:g}) centre=({row['mean_x1']:.6f},{row['mean_x2']:.6f}) "
              f"count={row['count']} ({100*row['fraction']:.2f}%) std=({row['std_x1']:.6f},{row['std_x2']:.6f})", flush=True)
    plot_comparison(target, trajectory[-1], labels, rows, args.figure)
    fig, update = reverse_frames(trajectory)
    frames = np.r_[np.arange(0, STEPS, args.frame_stride), STEPS]
    settings = {
        "checkpoint": str(args.checkpoint.resolve()), "samples": args.samples, "seed": args.seed, "ddpm_steps": STEPS,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02, "reverse_variance": "beta_t",
        "clipping": False, "final_step_noise": False, "initial_distribution": "N(0,I)",
        "target_centres": np.array(CENTRES).tolist(), "target_weights": [0.25] * 4,
        "target_component_variance": COMPONENT_VARIANCE, "mode_assignment": "nearest target centre",
        "variance_ddof": 0, "statistics_precision": "float64", "trajectory_precision": str(trajectory.dtype),
        "keys": {name: np.array(jax.random.key_data(key)).tolist() for name, key in (("sampling", sampling_key), ("target", target_key))},
        "prng": str(jax.random.key_impl(sampling_key)), "compute_seconds_including_compilation": compute_seconds,
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "figure": str(args.figure.resolve()), "movie": str(args.movie.resolve()), "fps": 30,
        "frame_levels": (STEPS - frames).tolist(), "coordinate_limits": list(fig.axes[0].get_xlim()),
        "colour_assignment": "fixed final nearest-mode labels", "codec": "libx264", "crf": 18, "pixel_format": "yuv420p",
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    movie = FuncAnimation(fig, update, frames=frames, interval=1000 / 30)
    args.movie.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=30, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(args.movie, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Saved {args.movie}, {args.figure} and raw results in {args.output}")


if __name__ == "__main__":
    main()
