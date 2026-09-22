#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
output="$root/artifacts/convergence"
target="${CARGO_TARGET_DIR:-$root/target}"
mkdir -p "$output"
cargo build --quiet --release --manifest-path "$root/Cargo.toml" --bin field --bin fluid
field="$target/release/field"
fluid="$target/release/fluid"

"$field" random --n 128 --seed 2026 --k-min 2 --k-max 6 > "$output/initial.json"
for dt in 0.02 0.0125 0.01 0.0025; do
    run="$output/rk4-dt$dt"
    mkdir -p "$run"
    "$fluid" --method rk4 --nu 0.004 --dt "$dt" --t-end 2 --every 2 \
        --out "$run" < "$output/initial.json" > "$run/diagnostics.tsv"
done
