# Week 7: Gaussian-mixture diffusion

Run from `week7/` with uv. CPU is the default; an explicit `JAX_PLATFORMS`
environment setting overrides it. Downloads belong in `inputs/`, raw runs with
settings in `artifacts/`, and figures in `evidence/`. These generated directories
are ignored by git.

```sh
uv run --no-project --with jax --with numpy --with matplotlib python scripts/run_diffusion.py
uv run --no-project --with jax --with numpy --with matplotlib --with pytest \
  python -m pytest scripts/test_diffusion.py -q
```

The forward run draws 4096 samples using seed 71. Change these with `--samples`
and `--seed`; `--output` and `--figure` select the destinations. It saves the full
Markov trajectory, levels and schedule in `artifacts/diffusion/forward.npz`,
settings, random keys, versions and device in `artifacts/diffusion/settings.json`,
and six snapshots in `evidence/forward-diffusion.png`.

## Forward process

The clean distribution is an equally weighted mixture with centres `(3,0)`,
`(0,3)`, `(-3,0)`, `(0,-3)` and component covariance `0.09 I` (standard deviation
`0.3`). Level `t=0` denotes clean samples; the 1000 transitions use

$$\beta_t = 0.0001 + \frac{t-1}{999}(0.02-0.0001),\quad t=1,\ldots,1000.$$

With $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$,

$$x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,z_t,\qquad z_t\sim N(0,I).$$

`scripts/diffusion.py` provides `sample_mixture(key, count)`,
`forward_trajectory(key, x0)` returning levels `0,...,1000`, and
`forward_sample(key, x0, t)` for a direct conditional marginal draw:

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

`BETA` and `ALPHA_BAR` are indexed by level and include the clean-state entries
`BETA[0]=0` and `ALPHA_BAR[0]=1`. The marginal remains a mixture with component
means $\sqrt{\bar\alpha_t}\mu_k$ and covariance
$(0.09\bar\alpha_t+1-\bar\alpha_t)I$.

The trajectory has shape `(1001, count, 2)` with `path[t]` holding level `t`.
The loop uses [JAX `lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
with a separate random key for each transition.

Tests check the target's equal weights, component means and covariances, direct
forward marginals, transition noise variances and independence from the previous
state, the terminal conditional distribution, and the saved-run command.

## Forward-diffusion movie

After generating the forward run above, create the movie with:

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.forward_movie
```

This requires the system `ffmpeg` executable. The movie is saved as
`evidence/forward-diffusion.mp4` (800 × 800, 30 fps, 1001 frames, about 33 seconds).
It follows the same 4096 samples through their saved Markov trajectories at every
level from `t=0` to `t=1000`. Colors are assigned by the nearest centre at `t=0`
and remain fixed, showing how the four initial modes merge. Both coordinate
axes share one fixed symmetric range, chosen from the entire saved trajectory
with padding so all samples remain visible.

The command prints each coordinate's empirical mean and population variance
(`ddof=0`) at `t=1000`. For the saved seed-71 run:

| Coordinate | Mean | Variance |
| --- | ---: | ---: |
| x1 | 0.006758 | 1.086408 |
| x2 | -0.000522 | 1.015046 |

`artifacts/diffusion/movie.json` records the source run, movie settings, fixed
coordinate limits, frame levels and terminal moments. `--run PATH`, `--output
PATH`, and `--fps N` select another saved run, movie destination or playback rate.
No fresh samples are drawn. The movie and its metadata remain outside git.
Tests check sample positions, persistent colors, fixed axes, known terminal
moments, encoded frame count and successful decoding using `ffmpeg`/`ffprobe`.

## Noise-prediction training and DDPM sampling

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.train_diffusion
uv run --no-project --with jax --with numpy --with matplotlib --with pytest \
  python -m pytest scripts -q
```

The predictor in `scripts/noise_model.py` has three hidden layers of width 128
with SiLU activations and a linear two-coordinate output (37,762 parameters).
Its input concatenates the noisy point with 16 sine and 16 cosine time features:

$$\sin[(t-1)10000^{-j/16}],\quad\cos[(t-1)10000^{-j/16}],\qquad j=0,\ldots,15.$$

Integer levels use the specified offset `-1`; time is not normalized. Each
weight is initialized independently from a normal distribution with standard
deviation `1/sqrt(input width)`; all biases start at zero. Computation uses
float32 and defaults to CPU.

At each update, independently draw 512 fresh target points, one uniform integer
level `t` from `1,...,1000` per point, and standard-normal noise `epsilon`.
Construct `x_t` from the forward marginal and minimize

$$L(\theta)=\operatorname{mean}_{n,d}
\left[\epsilon_\theta(x_t^{(n)},t_n)_d-\epsilon_d^{(n)}\right]^2.$$

Adam uses `beta1=0.9`, `beta2=0.999` and `epsilon=1e-8`. Seed 2026 is split
into separate initialization, training, evaluation, sampling and target-plot
keys. Training folds the zero-based update index into its key and then splits
into independent target, time and noise keys. For 25,000 updates, the applied
learning rate is

$$\eta_i=\frac{0.001}{2}\left[1+\cos\left(\frac{\pi i}{24999}\right)\right],
\qquad i=0,\ldots,24999.$$

The first update uses `0.001` and the final update uses exactly zero. Checkpoints
are retained after updates **0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000**;
update zero is the initialized model. Every retained model is evaluated on the
same independently drawn batch of 8192 examples, which is never used for updates.

The run writes these files in `artifacts/noise-model/`:

- `settings.json`: training and diffusion settings, architecture, keys, versions,
  precision, device, timings and generated-sample diagnostics.
- `loss.csv`: every update's applied learning rate and pre-update minibatch MSE.
- `stages.csv`: the ten checkpoint updates, filenames and fixed-batch evaluation MSE.
- `checkpoint-XXXXXX.npz`: all model weights and biases at the corresponding update.
- `samples.npz`: final-model DDPM samples and independent target points.

The figure `evidence/noise-training.png` shows training loss averaged in blocks
of 100 updates, the ten fixed-batch losses, and target/generated scatter plots.
Raw losses are retained without averaging. The output directory must be new to
avoid overwriting retained models; choose another with `--output PATH`.
`--figure`, `--samples`, `--seed`, `--batch-size`, `--learning-rate`, `--updates`
and `--checkpoint-updates` support explicit alternate runs. The ten checkpoint
updates must increase strictly from zero to the requested update count.

The reverse sampler starts at independent `N(0,I)` points and runs levels
`1000,...,1` using the trained predictor:

$$m_\theta(x_t,t)=\frac{x_t-\beta_t\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}}
{\sqrt{1-\beta_t}},$$

$$x_{t-1}=m_\theta(x_t,t)+\sqrt{\beta_t}z_t\quad(t>1),\qquad
x_0=m_\theta(x_1,1).$$

The reverse variance is `beta_t`. Neither predicted clean points nor reverse
states are clipped, and the final step adds no noise. This is the noise-prediction
sampling update from [Ho et al., DDPM](https://arxiv.org/abs/2006.11239) with the
specified reverse variance.

Reload a retained model and generate points from `week7/`:

```python
from scripts.noise_model import ddpm_trajectory
import jax
import jax.numpy as jnp
import numpy as np

with np.load("artifacts/noise-model/checkpoint-025000.npz") as saved:
    params = {name: jnp.array(saved[name]) for name in saved.files}
path = ddpm_trajectory(params, jax.random.key(2027), count=4096)
generated = path[-1]
```

Tests verify time features against an independent NumPy network calculation,
initialization scales, uniform training levels and actual added-noise targets,
cosine decay endpoints, reverse transition means and variances, and the
deterministic unclipped final step. A reduced real training run verifies loss
reduction, all ten reloadable checkpoints, records, figures, and generation
from the saved final weights.

The default seed-2026 CPU run reduced fixed-batch noise MSE from `1.001041`
at initialization to `0.254494` at update 25,000. Among 4096 generated points,
the nearest-centre fractions (in the centre order above) were `23.73%`, `26.15%`,
`24.98%`, and `25.15%`. Mean squared distance per coordinate to the nearest
centre was `0.09113`, compared with the target component variance `0.09`.
These are diagnostics from one run and finite sample, rather than a guarantee
that the learned distribution equals the target.

## Movie of saved training stages

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.training_movie
```

This reads the ten checkpoints listed in `artifacts/noise-model/stages.csv`
and writes `evidence/training-stages.mp4`. Each stage is shown for two seconds
(60 frames at 30 fps), for a 20-second movie at 1500 × 500 pixels. Every stage
shows ten complete DDPM paths from `t=1000` to `t=0` using its frozen predictor.
Path colors identify matching chains; `+` marks Gaussian starts and filled dots
mark generated endpoints. Grey circles mark the four target centres.

The first panel keeps both coordinate limits fixed at `[-5,5]`. The second uses
one fixed range containing every path at every stage, including the widely
spread paths of the initialized model. The original path coordinates are
retained and plotted without alteration.
The loss panel uses fixed axes and
shows raw training losses only through the current checkpoint, means of up to
100 consecutive losses, and held-out losses for the stages reached so far.
The training-update axis uses a symmetric log scale with a linear region through
update 50, making the early checkpoints visible alongside the final update.

All ten checkpoints use the training run's recorded sampling key and exactly ten
chains. This reuses identical initial Gaussian points and identical standard
normal reverse-step draws at every stage. The complete paths are retained in
`artifacts/noise-model/training-paths.npz`, with shape `(10,1001,10,2)` ordered
by checkpoint, decreasing diffusion level, chain and coordinate. The file also
records checkpoint updates and diffusion levels. `training-movie.json` records
the checkpoints, key, coupling convention, plot limits and encoding settings.
These raw runs and the movie remain outside git.

Use `--run PATH` and `--output PATH` for another saved run or movie destination;
`--frames-per-stage N` changes the time each checkpoint remains on screen.
The command requires system `ffmpeg`; the tests also use `ffprobe` to verify
encoding. Tests check shared random draws across different predictors, complete
path coordinates, stable colors, fixed plot limits, exclusion of future losses,
all ten saved-stage inputs, and successful movie decoding.

`ddpm_trajectory(params, key, count)` returns all 1001 levels, including the
initial Gaussian batch at index zero and generated points at index `-1`.
