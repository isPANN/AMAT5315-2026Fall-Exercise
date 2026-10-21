"""Compare paired deterministic DDIM runs and stochastic DDPM runs from one x_T."""

import argparse
import json
from pathlib import Path
import time

from scripts.diffusion import CENTRES, STEPS, sample_mixture
from scripts.noise_model import ddpm_trajectory, ddim_trajectory

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_runs(paths, exact_samples, output):
    names = ["ddim_first", "ddim_second", "ddpm_first", "ddpm_second", "ddim_exact"]
    titles = ["DDIM · run 1", "DDIM · run 2", "DDPM · noise draw 1", "DDPM · noise draw 2", "DDIM · exact noise predictor"]
    centres = np.array(CENTRES)
    limit = float(np.ceil(max(np.abs(exact_samples).max(), *(np.abs(path).max() for path in paths.values()))) + 1)
    colours = plt.get_cmap("tab10")(np.arange(10))
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), layout="constrained")
    for ax, name, title in zip(axes, names, titles):
        path = paths[name]
        for chain in range(min(10, path.shape[1])):
            ax.plot(path[:, chain, 0], path[:, chain, 1], color=colours[chain], linewidth=0.8)
        ax.scatter(path[0, :10, 0], path[0, :10, 1], c=colours[:path.shape[1]], marker="+", s=25)
        ax.scatter(path[-1, :10, 0], path[-1, :10, 1], c=colours[:path.shape[1]], s=15, edgecolors="black", linewidths=0.3)
        ax.scatter(centres[:, 0], centres[:, 1], s=65, facecolors="none", edgecolors="gray")
        ax.set(title=title, xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    fig.suptitle("Ten matched trajectories · same Gaussian starts · + start, filled dot finish")
    fig.savefig(output / "ddim-ddpm-trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    palette = np.array(["#0072b2", "#d55e00", "#009e73", "#cc79a7"])
    samples = [paths[name][-1] for name in names] + [exact_samples]
    for ax, points, title in zip(axes.flat, samples, titles + ["Exact samples · direct mixture draw"]):
        labels = ((points[:, None] - centres) ** 2).sum(-1).argmin(1)
        ax.scatter(points[:, 0], points[:, 1], c=palette[labels], s=4, alpha=0.35, linewidths=0)
        ax.scatter(centres[:, 0], centres[:, 1], marker="+", c="black", s=50)
        ax.set(title=title, xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    fig.suptitle(f"{len(exact_samples):,} final samples per panel · shared coordinate limits")
    fig.savefig(output / "ddim-ddpm-samples.png", dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "noise-model" / "checkpoint-025000.npz")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(STEPS, -1, -1)))
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "ddim-comparison")
    parser.add_argument("--figures", type=Path, default=root / "evidence")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    levels = tuple(args.levels)
    if len(levels) < 2 or levels[0] != STEPS or levels[-1] != 0 or any(t <= s for t, s in zip(levels[:-1], levels[1:])):
        parser.error("--levels must decrease strictly from 1000 to 0")
    with np.load(args.checkpoint) as saved:
        params = {name: jnp.array(saved[name]) for name in saved.files}
    args.output.mkdir(parents=True, exist_ok=False)
    keys = dict(zip(("initial", "ddpm_first", "ddpm_second", "target"), jax.random.split(jax.random.key(args.seed), 4)))
    initial = jax.random.normal(keys["initial"], (args.samples, 2), dtype=jnp.float32)
    started = time.perf_counter()
    paths = {
        "ddim_first": np.array(ddim_trajectory(params, initial, levels)),
        "ddim_second": np.array(ddim_trajectory(params, initial, levels)),
        "ddpm_first": np.array(ddpm_trajectory(params, initial, keys["ddpm_first"])),
        "ddpm_second": np.array(ddpm_trajectory(params, initial, keys["ddpm_second"])),
        "ddim_exact": np.array(ddim_trajectory(None, initial, levels, exact=True)),
    }
    elapsed = time.perf_counter() - started
    if any(not np.isfinite(path).all() for path in paths.values()):
        raise ValueError("sampler produced non-finite trajectories")
    exact_samples = np.array(sample_mixture(keys["target"], args.samples))
    difference = float(np.max(np.abs(paths["ddim_first"] - paths["ddim_second"])))
    ddpm_difference = float(np.max(np.abs(paths["ddpm_first"] - paths["ddpm_second"])))
    print(f"Largest DDIM difference (all trajectory coordinates): {difference:.9g}", flush=True)
    print(f"Largest DDPM difference (all trajectory coordinates): {ddpm_difference:.9g}", flush=True)
    np.savez_compressed(args.output / "runs.npz", **paths, initial=np.array(initial), exact_samples=exact_samples,
                        ddim_levels=np.array(levels), ddpm_levels=np.arange(STEPS, -1, -1))
    settings = {
        "checkpoint": str(args.checkpoint.resolve()), "samples": args.samples, "seed": args.seed,
        "ddim_levels": list(levels), "ddim_transitions": len(levels) - 1, "ddim_eta": 0,
        "ddpm_transitions": STEPS, "ddpm_reverse_variance": "beta_t", "clipping": False, "ddpm_final_step_noise": False,
        "coupling": "one explicit x_1000 shared by every trajectory; DDPM has independent reverse-noise keys",
        "exact_noise_reference": "mixture conditional-mean E[epsilon|x_t]; same DDIM levels and starting points",
        "exact_samples": "independent direct draws from the target mixture",
        "ddim_max_absolute_difference": difference, "ddpm_max_absolute_difference": ddpm_difference,
        "keys": {name: np.array(jax.random.key_data(key)).tolist() for name, key in keys.items()},
        "prng": str(jax.random.key_impl(keys["initial"])), "dtype": str(initial.dtype),
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "compute_seconds_including_compilation": elapsed,
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_runs(paths, exact_samples, args.figures)
    print(f"Saved runs and settings in {args.output}; trajectory and sample plots in {args.figures}")


if __name__ == "__main__":
    main()
