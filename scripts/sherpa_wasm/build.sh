#!/usr/bin/env bash
# Build the browser runtime of sherpa-onnx for the GigaAM endpoint pass (ADR D64).
#
#   scripts/sherpa_wasm/build.sh <work-dir> [<out-dir>]
#
# Needs: emsdk 4.0.23 activated (`source <emsdk>/emsdk_env.sh`; the version upstream documents
# as known to work), cmake >= 3.13 < 4, curl, shasum. Downloads the pinned sherpa-onnx release
# source (sha256-checked; its CMake fetches the pinned, hash-checked onnxruntime WASM static
# library), replaces wasm/web/CMakeLists.txt with ours (offline ASR only, single-threaded,
# WORKERFS) and writes sherpa-onnx-wasm-web.{js,wasm} plus upstream's sherpa-onnx-asr.js into
# <out-dir> (default: <work-dir>/out). Then print the sha256 of each file: they are pinned in
# src/qorgan/asr/gigaam_web.py, which deploy_bootstrap uses to verify the installed copy.
set -euo pipefail

SHERPA_VERSION="1.13.8"
SHERPA_SHA256="b0374cc56dbc186d442ae73d5de743bb092470b640c4c50ce7b029044c0c4fa8"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:?usage: build.sh <work-dir> [<out-dir>]}"
OUT="${2:-$WORK/out}"

command -v emcc >/dev/null || { echo "emcc not found: activate emsdk 4.0.23 first" >&2; exit 1; }
EMSCRIPTEN="$(dirname "$(realpath "$(command -v emcc)")")"
export EMSCRIPTEN

mkdir -p "$WORK" "$OUT"
cd "$WORK"
TARBALL="sherpa-onnx-$SHERPA_VERSION.tar.gz"
if [ ! -f "$TARBALL" ]; then
  curl -sSL -o "$TARBALL" "https://codeload.github.com/k2-fsa/sherpa-onnx/tar.gz/refs/tags/v$SHERPA_VERSION"
fi
echo "$SHERPA_SHA256  $TARBALL" | shasum -a 256 -c -
rm -rf "sherpa-onnx-$SHERPA_VERSION"
tar xzf "$TARBALL"
SRC="$WORK/sherpa-onnx-$SHERPA_VERSION"
cp "$HERE/CMakeLists.txt" "$SRC/wasm/web/CMakeLists.txt"
cp "$HERE/qorgan-sherpa.cc" "$SRC/wasm/web/qorgan-sherpa.cc"

mkdir -p "$SRC/build-qorgan-web"
cd "$SRC/build-qorgan-web"
export SHERPA_ONNX_IS_USING_BUILD_WASM_SH=ON
# sherpa-onnx logs with __FILE__: without the prefix map the absolute work directory is baked
# into the wasm and the hashes depend on where it was built (measured: two directories, two hashes).
# Every spelling of the source path is mapped: on macOS /tmp is /private/tmp, and the
# compiler sees whichever form CMake hands it.
REAL_SRC="$(realpath "$SRC")"
PREFIX_MAP="-ffile-prefix-map=$SRC=/sherpa-onnx -ffile-prefix-map=$REAL_SRC=/sherpa-onnx -ffile-prefix-map=${REAL_SRC#/private}=/sherpa-onnx"
cmake \
  -DCMAKE_C_FLAGS="$PREFIX_MAP" \
  -DCMAKE_CXX_FLAGS="$PREFIX_MAP" \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake" \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_GPU=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_WASM=ON \
  -DSHERPA_ONNX_ENABLE_WASM_WEB=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF \
  ..
make -j"$(getconf _NPROCESSORS_ONLN)" sherpa-onnx-wasm-web
make install >/dev/null

cp install/bin/wasm/web/sherpa-onnx-wasm-web.js install/bin/wasm/web/sherpa-onnx-wasm-web.wasm "$OUT/"
cp "$SRC/wasm/asr/sherpa-onnx-asr.js" "$OUT/"
cd "$OUT"
shasum -a 256 sherpa-onnx-wasm-web.js sherpa-onnx-wasm-web.wasm sherpa-onnx-asr.js
ls -l "$OUT"
