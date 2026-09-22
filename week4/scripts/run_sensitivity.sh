#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
output="$root/artifacts/sensitivity"
target="${CARGO_TARGET_DIR:-$root/target}"
mkdir -p "$output"
cargo build --quiet --release --manifest-path "$root/Cargo.toml" \
    --bin field --bin sensitivity-data
field="$target/release/field"
sensitivity="$target/release/sensitivity-data"

"$field" taylor-green --n 64 \
    | "$sensitivity" 0.1 "$output/taylor-green"
"$field" random --n 128 --seed 2026 --k-min 2 --k-max 6 \
    | "$sensitivity" 0.004 "$output/random"
