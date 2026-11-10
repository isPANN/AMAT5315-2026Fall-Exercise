# Gaussian matrix and truncated SVD

Run from this directory:

```sh
cargo run --release --locked --bin week10-svd
```

The generator is `rand_chacha` 0.3.1's `ChaCha20Rng`, initialized with `seed_from_u64(2026)`. `rand_distr` 0.4.3's `StandardNormal` samples the entries in nalgebra's column-major construction order. Exact dependency versions are recorded in `Cargo.lock`.

`results/matrix.bin` and `results/rank-{16,64,128,256,512}.bin` each contain a 512 × 512 matrix, stored row by row as little-endian IEEE 754 64-bit floats. `results/singular-values.bin` contains 512 such floats in descending order. Each rank-k matrix is the sum of the first k singular components. The output is generated locally and is not committed.

To plot the spectrum and independently recomputed errors, install NumPy and Matplotlib, then run `python3 plot.py` from this directory. The plot is saved to `artifacts/svd.png`. The storage marker counts two dense 512 × rank factors (not the singular values), equal to the 512 × 512 original at rank 256. The 10% line is a relative Frobenius-error threshold.

# Open Ising partition function

Run the boundary MPS solver from this directory, for example:

```sh
cargo run --release --locked --bin thermo -- --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
```

The output `evidence.json` records the settings, log partition function, per-site log partition function, solver time, row normalization scales, discarded squared singular values per bond and row, and the final normalized MPS. Tensor entries use row-major `(left, physical, right)` order. Contract each physical index with the recorded missing-bond vector and neighboring bond indices, then add the logarithms of the normalization scales to the logarithm of that final contraction to recover log Z. The discarded weights refer to the normalized incoming MPS and are not global error bounds.

Run `python3 critical.py` to reproduce the open-boundary sweep and `artifacts/boundary-critical.png`. It computes $c_V=K^2 d^2g/dK^2$ using centered differences, fits three neighboring specific-heat points as a parabola in temperature, and fits the χ=32 peak temperatures linearly in 1/L. The coarse and refined g tables include solver timings in `results/critical-*.csv`; `results/critical-peaks.json` records the fitted peaks and extrapolation. Full evidence for every refined L=64 run is in `runs/critical/`. Generated results remain local.

After the sweep, `python3 verify_critical.py` compares χ=16 and 32 peaks, repeats the χ=32 peak fits using every other refined sample, and checks the 4 × 4 partition function and specific heat against explicit spin enumeration. It writes the comparisons to `results/critical-verification.json` and fails if the requested tolerances are not met.

For a periodic square, use `--method trg` and a power-of-two `--l`, for example `cargo run --release --locked --bin thermo -- --method trg --l 4 --k 0.44 --chi 16 --out runs/trg4`. The implementation follows the two SVD splits and regrouping in the [BackwardsLinalg TRG example](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl), with max-absolute normalization and final closure `sum(A[i,j,i,j])`. Each step records the scale, tensor count, singular values and four split factors; the final tensor and its index order are also recorded. To recover log Z, sum `tensor_count * log(scale)` over steps and add the logarithm of the final closure. Periodic L=2 counts the two parallel bonds in each direction, as the tensor network does.

Run `python3 trg_critical.py` for the periodic TRG peak study. It saves coarse and refined g tables and timings in `results/trg-critical-*.csv`, the rank, grid-spacing and size-fit comparisons in `results/trg-critical-summary.json`, and the figure in `artifacts/trg-critical.png`. Full refined L=1024 run evidence is retained under `runs/trg-critical/`. Independent points are computed concurrently, so reported timings reflect the shared machine load.

After that sweep, `python3 verify_trg.py` compares three-size and two-largest-size extrapolations and checks periodic L=2 and L=4 against spin enumeration, including the parallel bonds at L=2. It records the fit-window and enumeration checks in `results/trg-critical-verification.json`.
