# Gaussian matrix and truncated SVD

Run from this directory:

```sh
cargo run --release --locked
```

The generator is `rand_chacha` 0.3.1's `ChaCha20Rng`, initialized with `seed_from_u64(2026)`. `rand_distr` 0.4.3's `StandardNormal` samples the entries in nalgebra's column-major construction order. Exact dependency versions are recorded in `Cargo.lock`.

`results/matrix.bin` and `results/rank-{16,64,128,256,512}.bin` each contain a 512 × 512 matrix, stored row by row as little-endian IEEE 754 64-bit floats. `results/singular-values.bin` contains 512 such floats in descending order. Each rank-k matrix is the sum of the first k singular components. The output is generated locally and is not committed.

To plot the spectrum and independently recomputed errors, install NumPy and Matplotlib, then run `python3 plot.py` from this directory. The plot is saved to `artifacts/svd.png`. The storage marker counts two dense 512 × rank factors (not the singular values), equal to the 512 × 512 original at rank 256. The 10% line is a relative Frobenius-error threshold.
