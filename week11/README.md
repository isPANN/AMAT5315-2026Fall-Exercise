# Week 11: tensor-network sampling

`tnsample` enumerates spin gadgets, plans tensor-network contractions, counts
spin-glass and N-queens configurations, and samples from a stored contraction
tree. Commands below run from `week11/`. Generated problem instances, run
evidence, build outputs, and figures stay local and are ignored by Git.

## Environment and setup

The recorded runs used macOS on Apple Silicon (`Darwin arm64`), `rustc 1.98.0`
(88d9e12ae, 2026-08-18), `cargo 1.98.0` (797e8a9bc, 2026-08-05),
`pred 0.6.0`, Python 3.15.0, and gnuplot 6.0 patchlevel 5 with the
`pngcairo` terminal. `pred` must be installed separately and available on
`PATH`. Rust dependencies, including omeco 0.2.6, are pinned in `Cargo.lock`.

```sh
cargo build --release --locked
cargo test --locked
tnsample_bin=./target/release/tnsample
mkdir -p runs/pred
```

Every run below uses seed **2026**. The seed fixes lattice bonds and drives
sampling; omeco may choose a different contraction tree across runs, so the
same seed does not guarantee identical sample sequences or planner metrics.
Each `run.json` records the run parameters, planner tree, and raw evidence.

## Gadgets

```sh
"$tnsample_bin" --mode gadgets --seed 2026 --out runs/gadgets
python3 plot_gates.py
```

This creates `artifacts/gates.png`. Bits use `s = 2x - 1`; XOR minimizes
over its helper spin. The plot script independently runs the enumeration and
saves its temporary evidence under `target/plot-run/`.

## Factoring

Generate the `pred` SpinGlass bundles and plan both contraction orders for
`k = 2..8`. Each scale instance has two `k`-bit factors and target
`(2^k - 1)^2`. The sample instance factors 15 with factor widths 2 and 3.

```sh
for k in 2 3 4 5 6 7 8; do
    target=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
    pred create Factoring --m "$k" --n "$k" --target "$target" -o "runs/pred/scale_$k.json"
    pred reduce "runs/pred/scale_$k.json" --to SpinGlass -o "runs/pred/scale_$k.bundle.json"
    for order in greedy treesa; do
        "$tnsample_bin" --mode plan --input "runs/pred/scale_$k.bundle.json" \
            --order "$order" --seed 2026 --out "runs/plan_k${k}_${order}"
    done
done
pred create Factoring --m 2 --n 3 --target 15 -o runs/pred/factor_15.json
pred reduce runs/pred/factor_15.json --to SpinGlass -o runs/pred/factor_15.bundle.json
"$tnsample_bin" --mode count --input runs/pred/factor_15.bundle.json \
    --beta 2.0 --order treesa --seed 2026 --out runs/count_f15
"$tnsample_bin" --mode sample --input runs/pred/factor_15.bundle.json \
    --beta 2.0 --samples 2000 --order treesa --seed 2026 --out runs/sample_f15
python3 plot_factoring.py
```

The script writes `artifacts/factoring.png`, compares the ground-state sample
count with `2000 * g_0 * exp(-beta * E_min) / Z`, and runs `pred extract` on a
lowest-energy sample. The recorded run found factors 3 and 5. Its SpinGlass
bundle uses `+sum J_ij s_i s_j + sum h_i s_i` for energy; `E_min` and `g_0`
come from min-plus contraction, and `log Z` comes from log-scaled tensors.

For an open square lattice instead, specify `--l`, `--p-af`, and `--beta` in
place of `--input`. The recorded ferromagnetic lattice count used:

```sh
"$tnsample_bin" --mode count --l 10 --p-af 0 --beta 2.0 \
    --order treesa --seed 2026 --out runs/count_p0
```

Lattice energy is `-sum J_ij s_i s_j`, with antiferromagnetic bonds (`J = -1`)
drawn with probability `p-af`. The saved edges use the bundle's positive-sum
convention. `--order` selects `greedy` or `treesa` (default `treesa`).

## N-queens

Count the carry network for N = 4 through 8 and draw 200 boards for N = 6 and
N = 8. The plotting script checks the known counts and every saved board in
`runs/sample_nq*/run.json`, then plots N = 6 frequencies.

```sh
for n in 4 5 6 7 8; do
    "$tnsample_bin" --mode count --n "$n" --order treesa --seed 2026 --out "runs/count_nq$n"
done
for n in 6 8; do
    "$tnsample_bin" --mode sample --n "$n" --samples 200 --order treesa \
        --seed 2026 --out "runs/sample_nq$n"
done
python3 plot_nqueens.py
```

This produces `artifacts/nqueens.png`. Each sampled board is an N by N array
of 0/1 entries, with 1 for a queen; plot labels list the queen column in each
row, numbered from 1. Exact counts use arbitrary-precision integers; requesting
samples fails when there are no solutions. N-queens contraction stores only
nonzero entries, while reported `tc` and `sc` are omeco's dense-network
estimates, not measurements of sparse time or memory.
