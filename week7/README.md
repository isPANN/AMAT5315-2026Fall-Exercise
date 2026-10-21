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
reverse_key, initial_key = jax.random.split(jax.random.key(2027))
initial = jax.random.normal(initial_key, (4096, 2), dtype=jnp.float32)
path = ddpm_trajectory(params, initial, reverse_key)
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

`ddpm_trajectory(params, initial, key)` returns all 1001 levels, including the
explicit starting batch at index zero and generated points at index `-1`.
The key controls only reverse-step noise.

## Reverse-process movie and final distribution

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.reverse_movie
```

This loads `checkpoint-025000.npz`, draws 2000 independent Gaussian starting
points, and performs all 1000 DDPM reverse steps with the frozen trained
predictor. Reverse variance is `beta_t`, with no clipping and no added noise in
the last step. Seed 2027 is split into independent sampling and target-draw keys.
The run uses CPU by default.

`evidence/reverse-diffusion.mp4` follows the same points at every level from
`t=1000` to `t=0`, at 30 fps (1001 frames, about 33 seconds). Both coordinate
axes keep the same scale throughout, covering the whole trajectory. Colors
identify each point's final nearest target mode and stay fixed during playback.
`evidence/reverse-distribution.png` compares the final generated points with
2000 independently sampled target points on identical axes, overlays target
and measured centres, and includes the measured mode statistics.

Raw outputs are retained in `artifacts/reverse-diffusion/`:

- `trajectory.npz`: all 1001 states, decreasing diffusion levels, and target points.
- `mode-statistics.csv`: nearest-target-centre assignments summarized by mode,
  including count, fraction, centre, coordinate standard deviations and variances.
- `settings.json`: checkpoint path, sample count, seeds/keys, sampler and movie
  settings, precision, device, library versions and generation time.

Standard deviations and variances are population moments (`ddof=0`), computed
in float64 about each assigned group's measured centre. The target has 25%
probability per mode (500 points in expectation) and coordinate standard
deviations `0.3` (variances `0.09`). The default generated sample has:

| Target centre | Measured centre | Population | Std x1 | Std x2 |
| --- | --- | ---: | ---: | ---: |
| (3,0) | (2.9954,-0.0181) | 508 (25.40%) | 0.3026 | 0.3022 |
| (0,3) | (0.0016,2.9898) | 509 (25.45%) | 0.2925 | 0.3181 |
| (-3,0) | (-3.0062,0.0274) | 500 (25.00%) | 0.3029 | 0.2963 |
| (0,-3) | (0.0072,-2.9945) | 483 (24.15%) | 0.3221 | 0.3161 |

These estimates describe this one 2000-point run. Use `--checkpoint`, `--samples`
and `--seed` to select another generation run, and `--output`, `--figure` and
`--movie` to select destinations. The raw output directory must be new.
`--frame-stride N` subsamples movie frames while keeping the complete 1000-step
sampling computation and both movie endpoints. The tests use this to verify
the actual saved run, statistics, plot and video with a shorter encoding.
System `ffmpeg` is required; tests also use `ffprobe`.

## Deterministic DDIM and matched DDPM comparisons

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.compare_samplers
```

The default CPU run loads the final trained checkpoint and draws one batch of
2000 Gaussian starting points using seed 2028. Two DDIM runs use exactly this
same array and all 1000 positive noise levels, followed by the clean level zero.
Two DDPM runs start from that array and use independent reverse-noise keys.
The run also evaluates DDIM with the mixture's exact conditional-mean noise
predictor and draws 2000 independent exact target samples.

`evidence/ddim-ddpm-trajectories.png` shows ten matching trajectories per run,
with colors identifying the same starting point across all panels.
`evidence/ddim-ddpm-samples.png` shows 2000 final samples from each run alongside
direct target draws, using shared coordinate limits. The full trajectories,
initial points, direct target samples and selected levels are saved in
`artifacts/ddim-comparison/runs.npz`; `settings.json` records the checkpoint,
random keys, sampler settings, versions, device and measured differences.

The largest absolute difference between the paired DDIM runs is **0.0**, measured
over every coordinate of all 2000 samples at all 1001 states. The paired DDPM
runs have largest absolute difference **8.244673** over the same trajectory
coordinates. These are comparisons of paired paths, rather than distribution
distance measures.

For levels `t > s`, the deterministic DDIM update is

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon(x_t,t)}{\sqrt{\bar\alpha_t}},
\qquad x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\hat\epsilon(x_t,t).$$

This is the `eta=0` update from [Song et al., DDIM](https://arxiv.org/abs/2010.02502).
There is no random draw or clipping in a DDIM step. The trained network supplies
the default noise prediction. For the analytic comparison, `exact_noise(x,t)`
uses the forward marginal mixture: component means are
`sqrt(alpha_bar[t])*centre`, common variance is `v=0.09*alpha_bar[t]+1-alpha_bar[t]`,
and component probabilities are normalized Gaussian density weights `w_k`.
The conditional-mean noise is `sqrt(1-alpha_bar[t])*(x-sum(w_k*mean_k))/v`.
An exact conditional-mean predictor still undergoes finite-step DDIM sampling;
the direct target draws are a separate reference distribution.

Select a smaller DDIM level subset explicitly, ending with zero:

```sh
uv run --no-project --with jax --with numpy --with matplotlib python -m scripts.compare_samplers \
  --levels 1000 800 500 200 100 10 1 0 --output artifacts/ddim-subset --figures evidence/ddim-subset
```

DDPM always uses all 1000 reverse transitions in this comparison. `--samples`,
`--seed`, `--checkpoint`, `--output` and `--figures` select alternate runs and
destinations; the raw output directory must be new. Direct Python use accepts a
strictly decreasing tuple of integer levels, whose first entry lies in `1..1000`
and whose last entry is zero:

```python
from scripts.noise_model import ddim_trajectory

levels = (1000, 800, 500, 200, 100, 10, 1, 0)
learned_path = ddim_trajectory(params, initial, levels)
analytic_path = ddim_trajectory(None, initial, levels, exact=True)
```

The supplied `initial` points represent the first selected level. DDIM returns
one state per selected level. Its tests check skipped-level updates, deterministic
repetition, valid level sequences, the analytic noise against an independent
log-density gradient, mode recovery with that predictor, explicit shared DDPM
starts with independent step noise, and the saved comparison workflow.
