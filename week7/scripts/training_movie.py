"""Compare ten saved predictors using matched DDPM paths and loss history."""

import argparse
import csv
import json
from pathlib import Path

from scripts.diffusion import CENTRES, STEPS
from scripts.noise_model import ddpm_trajectory

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def training_frames(paths, updates, history, evaluation):
    """Show complete matched paths and only losses known at the current stage."""
    colours = plt.get_cmap("tab10")(np.arange(10))
    limit = float(np.ceil(max(np.max(np.abs(paths)), 3.)) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.26, top=0.78, wspace=0.3)
    path_artists = []
    for ax, bound, title in zip(axes[:2], (5., limit), ("Fixed target-scale view", "Full-path overview")):
        ax.scatter(np.array(CENTRES)[:, 0], np.array(CENTRES)[:, 1], s=100,
                   facecolors="none", edgecolors="#777777", linewidths=1.2)
        lines = [ax.plot([], [], color=colours[i], linewidth=0.8, alpha=0.75)[0] for i in range(10)]
        starts = ax.scatter(paths[0, 0, :, 0], paths[0, 0, :, 1], c=colours, marker="+", s=45, linewidths=1.4)
        ends = ax.scatter(paths[0, -1, :, 0], paths[0, -1, :, 1], c=colours, s=32, edgecolors="black", linewidths=0.4, zorder=5)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal", title=title,
               xlabel="$x_1$", ylabel="$x_2$")
        path_artists.append((lines, starts, ends))
    raw, = axes[2].plot([], [], color="#aaaaaa", linewidth=0.5, alpha=0.4, label="raw training MSE")
    averaged, = axes[2].plot([], [], ".-", color="#0072b2", linewidth=1.5, markersize=3, label="means of up to 100 updates")
    held_out, = axes[2].plot([], [], "o", color="#d55e00", markersize=4, label="fixed held-out batch")
    axes[2].set(xlim=(0, updates[-1]), ylim=(0, 1.08 * max(history[:, 1].max(), evaluation.max())),
                xlabel="Training update (symmetric log scale)", ylabel="Noise MSE", title="Loss history so far")
    axes[2].set_xscale("symlog", linthresh=50)
    axes[2].set_xticks([0, 50, 200, 1000, 5000, 25000], labels=["0", "50", "200", "1k", "5k", "25k"])
    axes[2].legend(fontsize=8, loc="upper right")
    title = fig.suptitle("")
    fig.text(0.5, 0.11, "10 complete paths: t = 1000 → 0 · + Gaussian start · ● generated endpoint · grey circles: target centres",
             ha="center", fontsize=10)
    fig.text(0.5, 0.065, "Matching colours reuse identical initial points and reverse-step noise at every stage", ha="center", fontsize=10)

    def update(stage):
        title.set_text(f"Saved training stage {stage + 1}/10 · update {updates[stage]:,}\nHeld-out noise MSE: {evaluation[stage]:.6f}")
        for lines, starts, ends in path_artists:
            for chain, line in enumerate(lines):
                points = paths[stage, :, chain]
                line.set_data(points[:, 0], points[:, 1])
            starts.set_offsets(paths[stage, 0])
            ends.set_offsets(paths[stage, -1])
        prefix = history[history[:, 0] <= updates[stage]]
        raw.set_data(prefix[:, 0], prefix[:, 1])
        blocks = [prefix[start:start + 100] for start in range(0, len(prefix), 100)]
        averaged.set_data([block[-1, 0] for block in blocks], [block[:, 1].mean() for block in blocks])
        held_out.set_data(updates[:stage + 1], evaluation[:stage + 1])
        return (title, raw, averaged, held_out)

    update(0)
    return fig, update


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "artifacts" / "noise-model")
    parser.add_argument("--output", type=Path, default=root / "evidence" / "training-stages.mp4")
    parser.add_argument("--frames-per-stage", type=int, default=60)
    args = parser.parse_args()
    if args.frames_per_stage < 1:
        parser.error("--frames-per-stage must be positive")
    with (args.run / "stages.csv").open() as stream:
        stages = list(csv.DictReader(stream))
    updates = np.array([int(stage["update"]) for stage in stages])
    if len(stages) != 10 or updates[0] != 0 or np.any(np.diff(updates) <= 0):
        raise ValueError("need ten saved training stages starting at initialization, in increasing order")
    evaluation = np.array([float(stage["evaluation_loss"]) for stage in stages])
    with (args.run / "loss.csv").open() as stream:
        history = np.array([(int(row["update"]), float(row["training_loss"])) for row in csv.DictReader(stream)])
    settings = json.loads((args.run / "settings.json").read_text())
    key = jax.random.wrap_key_data(jnp.array(settings["keys"]["sampling"], dtype=jnp.uint32))
    trajectories = []
    for stage in stages:
        with np.load(args.run / stage["checkpoint"]) as saved:
            params = {name: jnp.array(saved[name]) for name in saved.files}
        # Reusing the same key and shape couples every random draw across stages.
        trajectories.append(np.array(ddpm_trajectory(params, key, 10)))
        print(f"Sampled 10 matched paths at update {stage['update']}", flush=True)
    paths = np.stack(trajectories)
    if not np.isfinite(paths).all():
        raise ValueError("saved predictor produced non-finite generation paths")
    np.savez_compressed(args.run / "training-paths.npz", trajectories=paths, updates=updates, levels=np.arange(STEPS, -1, -1))
    fig, update = training_frames(paths, updates, history, evaluation)
    movie = FuncAnimation(fig, lambda frame: update(frame // args.frames_per_stage),
                          frames=10 * args.frames_per_stage, interval=1000 / 30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=30, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(args.output, writer=writer, dpi=100)
    metadata = {
        "run": str(args.run.resolve()), "output": str(args.output.resolve()),
        "checkpoints": [stage["checkpoint"] for stage in stages], "updates": updates.tolist(),
        "paths_per_stage": 10, "sampling_key": np.array(jax.random.key_data(key)).tolist(),
        "coupling": "same ddpm_trajectory key and sample count at every stage; no stage-dependent keys",
        "key_rule": "split once for initialization; split carried key at each reverse step",
        "fps": 30, "frames_per_stage": args.frames_per_stage,
        "target_view_limits": list(fig.axes[0].get_xlim()), "overview_limits": list(fig.axes[1].get_xlim()),
        "loss_display": "only updates <= checkpoint; raw losses plus non-overlapping means of up to 100 updates",
        "backend": jax.default_backend(), "dtype": str(paths.dtype), "jax": jax.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "reverse_variance": "beta_t", "clipping": False, "final_step_noise": False,
        "codec": "libx264", "crf": 18, "pixel_format": "yuv420p",
    }
    (args.run / "training-movie.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
