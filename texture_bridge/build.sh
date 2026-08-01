#!/bin/bash
# Build the mlx-tomo hardware-texture bridge dylib (optional backend).
set -euo pipefail
cd "$(dirname "$0")"
clang -dynamiclib -fobjc-arc -O2 \
  -framework Metal -framework Foundation \
  bridge.m -o libmlxtomo_bridge.dylib
cp libmlxtomo_bridge.dylib ../mlx_tomo/
echo "built $(pwd)/libmlxtomo_bridge.dylib (copied into mlx_tomo/)"
