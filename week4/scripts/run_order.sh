#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
order="$root/artifacts/order"
target="${CARGO_TARGET_DIR:-$root/target}"
mkdir -p "$order"
cargo build --quiet --release --manifest-path "$root/Cargo.toml" --bin field --bin fluid
field="$target/release/field"
fluid="$target/release/fluid"

"$field" taylor-green --n 8 --nu 0.5 --t 2 > "$order/exact-t2.json"
for dt in 0.4 0.25 0.2; do
    run="$order/rk4-dt$dt"
    mkdir -p "$run"
    "$field" taylor-green --n 8 \
        | "$fluid" --method rk4 --nu 0.5 --dt "$dt" --t-end 2 --every 2 \
            --out "$run" > "$run/diagnostics.tsv"
done
