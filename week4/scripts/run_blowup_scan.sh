#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
scan="$root/artifacts/scan"
target="${CARGO_TARGET_DIR:-$root/target}"
mkdir -p "$scan"
cargo build --quiet --release --manifest-path "$root/Cargo.toml" --bin field --bin fluid
field="$target/release/field"
fluid="$target/release/fluid"

run_taylor() {
    local dt=$1 name="taylor-rk4-$1"
    set +e
    "$field" taylor-green --n 64 \
        | "$fluid" --method rk4 --nu 0.1 --dt "$dt" --t-end 8 --every 0.5 \
            --out "$scan/$name" > "$scan/$name.tsv"
    local status=(${PIPESTATUS[@]})
    set -e
    [[ ${status[0]} -eq 0 && ${status[1]} -le 1 ]] || return 2
    printf '%s\tfluid exit %s\n' "$name" "${status[1]}"
}

run_random() {
    local method=$1 dt=$2 name="random-$1-$2"
    set +e
    "$field" random --n 128 --seed 2026 --k-min 2 --k-max 6 \
        | "$fluid" --method "$method" --nu 0.004 --dt "$dt" --t-end 10 --every 0.5 \
            --out "$scan/$name" > "$scan/$name.tsv"
    local status=(${PIPESTATUS[@]})
    set -e
    [[ ${status[0]} -eq 0 && ${status[1]} -le 1 ]] || return 2
    printf '%s\tfluid exit %s\n' "$name" "${status[1]}"
}

"$field" random --n 128 --seed 2026 --k-min 2 --k-max 6 > "$scan/random-initial.json"
run_taylor 0.032
run_taylor 0.033
run_random rk4 0.038
run_random rk4 0.040
run_random euler 0.01
