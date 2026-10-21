"""Train a time-conditioned noise predictor, retain ten stages, and sample DDPM."""

import argparse
import csv
import json
from pathlib import Path
import platform
import time

from scripts.diffusion import CENTRES, COMPONENT_VARIANCE, STEPS, sample_mixture
from scripts.noise_model import ddpm_trajectory, initialize_model, learning_rate, noise_loss, training_batch

import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jaxlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-updates", type=int, nargs=10,
                        default=[0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "noise-model")
    parser.add_argument("--figure", type=Path, default=root / "evidence" / "noise-training.png")
    args = parser.parse_args()
    if args.updates < 9 or args.batch_size < 1 or args.samples < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("need at least 9 updates, positive batch/sample counts, and a finite positive learning rate")
    stages = np.array(args.checkpoint_updates)
    if stages[0] != 0 or stages[-1] != args.updates or np.any(np.diff(stages) <= 0):
        parser.error("the 10 checkpoint updates must increase strictly from 0 to --updates")
    args.output.mkdir(parents=True, exist_ok=False)

    keys = dict(zip(("initialization", "training", "evaluation", "sampling", "target"),
                    jax.random.split(jax.random.key(args.seed), 5)))
    params = initialize_model(keys["initialization"])
    settings = {
        "updates": args.updates, "batch_size": args.batch_size, "seed": args.seed,
        "hidden_widths": [128, 128, 128], "activation": "SiLU", "parameter_count": sum(p.size for p in params.values()),
        "time_features": 32, "time_frequency_base": 10000., "time_offset": -1,
        "time_encoding": "sin and cos of (t-1)*10000**(-j/16), j=0..15; no time normalization",
        "weight_initialization": "normal with std 1/sqrt(input width)", "bias_initialization": 0.,
        "optimizer": "Adam", "adam_b1": 0.9, "adam_b2": 0.999, "adam_epsilon": 1e-8,
        "learning_rate_initial": args.learning_rate, "learning_rate_final": 0.,
        "learning_rate_decay": "cosine",
        "learning_rate_schedule": "initial/2*(1+cos(pi*index/(updates-1))), index=0..updates-1",
        "loss": "mean squared epsilon error over batch and both coordinates",
        "training_data": "fresh independent target samples each update",
        "noise_levels": "independent discrete uniform integers 1..1000 per sample",
        "evaluation_samples": 8192, "evaluation": "same independent held-out batch at all 10 stages",
        "stage_updates": stages.tolist(), "diffusion_steps": STEPS,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "centres": np.array(CENTRES).tolist(), "weights": [0.25] * 4,
        "component_covariance": (COMPONENT_VARIANCE * np.eye(2)).tolist(),
        "samples": args.samples, "reverse_variance": "beta_t", "clipping": False,
        "final_step_noise": False, "sampling_initialization": "N(0,I)",
        "keys": {name: np.array(jax.random.key_data(key)).tolist() for name, key in keys.items()},
        "training_key_rule": "fold_in(training_key, zero-based update index); split into data/time/noise keys",
        "prng": str(jax.random.key_impl(keys["training"])), "dtype": str(params["w0"].dtype),
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    }
    settings_path = args.output / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    evaluation = training_batch(keys["evaluation"], settings["evaluation_samples"])
    evaluate = jax.jit(noise_loss)
    opt_init, opt_update, get_params = optimizers.adam(
        lambda i: learning_rate(i, args.updates, args.learning_rate))
    state = opt_init(params)

    @jax.jit
    def train_block(state, indices):
        def step(state, i):
            batch = training_batch(jax.random.fold_in(keys["training"], i), args.batch_size)
            loss, gradient = jax.value_and_grad(noise_loss)(get_params(state), *batch)
            return opt_update(i, gradient, state), loss
        return jax.lax.scan(step, state, indices)

    evaluation_losses, training_losses = [], []
    started = time.perf_counter()
    with (args.output / "loss.csv").open("w", newline="") as loss_file, (args.output / "stages.csv").open("w", newline="") as stage_file:
        loss_writer, stage_writer = csv.writer(loss_file), csv.writer(stage_file)
        loss_writer.writerow(["update", "learning_rate", "training_loss"])
        stage_writer.writerow(["stage", "update", "evaluation_loss", "checkpoint"])
        for stage, update in enumerate(stages):
            if stage:
                indices = np.arange(stages[stage - 1], update)
                state, losses = train_block(state, jnp.array(indices))
                losses = np.array(losses)
                rates = np.array(learning_rate(jnp.array(indices), args.updates, args.learning_rate))
                training_losses.extend(losses.tolist())
                loss_writer.writerows((int(i + 1), float(rate), float(loss))
                                      for i, rate, loss in zip(indices, rates, losses))
                loss_file.flush()
            params = get_params(state)
            value = float(evaluate(params, *evaluation))
            evaluation_losses.append(value)
            checkpoint = f"checkpoint-{update:06d}.npz"
            np.savez_compressed(args.output / checkpoint, **{name: np.array(p) for name, p in params.items()})
            stage_writer.writerow([stage, update, value, checkpoint])
            stage_file.flush()
            print(f"stage={stage} update={update} evaluation_mse={value:.6f}", flush=True)
    settings["training_and_evaluation_seconds_including_compilation"] = time.perf_counter() - started

    # Generate from reloaded final weights to verify the retained model is usable.
    with np.load(args.output / checkpoint) as saved:
        final = {name: jnp.array(saved[name]) for name in saved.files}
    started = time.perf_counter()
    reverse_key, initial_key = jax.random.split(keys["sampling"])
    initial = jax.random.normal(initial_key, (args.samples, 2), dtype=jnp.float32)
    generated = np.array(ddpm_trajectory(final, initial, reverse_key)[-1])
    settings["sampling_seconds_including_compilation"] = time.perf_counter() - started
    target = np.array(sample_mixture(keys["target"], args.samples))
    np.savez_compressed(args.output / "samples.npz", generated=generated, target=target)
    labels = np.argmin(np.sum((generated[:, None] - np.array(CENTRES)) ** 2, axis=-1), axis=1)
    settings["generated_mode_fractions"] = (np.bincount(labels, minlength=4) / len(generated)).tolist()
    settings["generated_mean_squared_distance_to_nearest_centre"] = float(np.mean((generated - np.array(CENTRES)[labels]) ** 2))
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    losses = np.array(training_losses)
    # Average adjacent updates only for display; loss.csv retains every raw loss.
    block = min(100, len(losses))
    ends = np.arange(block, len(losses) + 1, block)
    axes[0].plot(ends, [losses[end - block:end].mean() for end in ends], label=f"training ({block}-update means)", alpha=0.75)
    axes[0].plot(stages, evaluation_losses, "o-", label="fixed held-out batch")
    axes[0].set(xlabel="Training update", ylabel="Noise MSE", title="Noise-prediction loss")
    axes[0].legend(fontsize=8)
    for ax, points, title in zip(axes[1:], (target, generated), ("Target mixture", "Trained DDPM samples")):
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.3, color="#266c8a")
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    limit = max(5., float(np.max(np.abs(np.concatenate((target, generated))))) * 1.05)
    for ax in axes[1:]:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(f"Saved 10 models, loss history, settings and samples in {args.output}; figure {args.figure}")


if __name__ == "__main__":
    main()
