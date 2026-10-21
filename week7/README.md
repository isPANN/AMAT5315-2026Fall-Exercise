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
