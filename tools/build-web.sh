#!/usr/bin/env bash
# Build the browser bundle: the WebAssembly module plus its generated parameter
# table. Everything else in web/ is hand-written and needs no build step.
set -euo pipefail
cd "$(dirname "$0")/.."

# simd128 is supported by every browser that supports WebAssembly threads and
# then some; it is a straight win on the brain and field loops.
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release -p borscht-wasm --target wasm32-unknown-unknown

cp target/wasm32-unknown-unknown/release/borscht_wasm.wasm web/borscht.wasm
cargo run --release -q -p borscht-cli -- params --json > web/params.js

printf 'web/borscht.wasm  %s KiB\n' "$(( $(stat -c%s web/borscht.wasm) / 1024 ))"
printf 'web/params.js     %s parameters\n' "$(grep -c 'id:' web/params.js)"
echo
echo 'Serve the directory over HTTP (ES modules and WebAssembly need it):'
echo '  python3 -m http.server -d web 8080'
