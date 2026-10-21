"""Save a CPU target-sampling and forward-diffusion run with its settings."""

import argparse
import json
from pathlib import Path
import platform
import time

from diffusion import ALPHA_BAR, BETA, CENTRES, COMPONENT_VARIANCE, STEPS, forward_trajectory, sample_mixture

import jax
import jaxlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "diffusion")
    parser.add_argument("--figure", type=Path, default=root / "evidence" / "forward-diffusion.png")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    mixture_key, forward_key = jax.random.split(jax.random.key(args.seed))
    started = time.perf_counter()
    x0 = sample_mixture(mixture_key, args.samples)
    trajectory = np.array(forward_trajectory(forward_key, x0))
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "forward.npz", trajectory=trajectory,
                        levels=np.arange(STEPS + 1), beta=np.array(BETA), alpha_bar=np.array(ALPHA_BAR))
    settings = {
        "samples": args.samples, "seed": args.seed, "steps": STEPS,
        "centres": np.array(CENTRES).tolist(), "weights": [0.25] * 4,
        "component_covariance": (COMPONENT_VARIANCE * np.eye(2)).tolist(),
        "schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "level_order": "0 to 1000", "dtype": str(trajectory.dtype),
        "keys": {name: np.array(jax.random.key_data(key)).tolist()
                 for name, key in (("mixture", mixture_key), ("forward", forward_key))},
        "prng": str(jax.random.key_impl(mixture_key)),
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "compute_seconds_including_compilation": elapsed,
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    levels = [0, 100, 250, 500, 750, 1000]
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 3.3), layout="constrained")
    for ax, t in zip(axes, levels):
        points = trajectory[t]
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.25, color="#266c8a", rasterized=True)
        ax.set(title=f"t = {t}", xlim=(-5, 5), ylim=(-5, 5), xlabel="$x_1$", aspect="equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(f"Forward diffusion · {args.samples:,} samples · 1000 linear-variance steps")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output / 'forward.npz'}, {args.output / 'settings.json'} and {args.figure}")


if __name__ == "__main__":
    main()
