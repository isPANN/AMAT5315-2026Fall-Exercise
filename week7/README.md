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
from scripts.noise_model import ddpm_sample
import jax
import jax.numpy as jnp
import numpy as np

with np.load("artifacts/noise-model/checkpoint-025000.npz") as saved:
    params = {name: jnp.array(saved[name]) for name in saved.files}
generated = ddpm_sample(params, jax.random.key(2027), count=4096)
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
