#!/usr/bin/env bash
set -euo pipefail

# Static arm64 + Metal (embedded shaders) build of the CrispASR C API for the
# Context-Assist voice-worker. Produces one merged archive so the consumer
# links `static=crispasr` plus system frameworks and ships no CrispASR/ggml
# dylibs (the app bundle already carries llama.cpp's libggml*.dylib).
#
# Usage: context-assist/build-static.sh <out-dir>
# Result: <out-dir>/crispasr-<tag>-macos-arm64.tar.gz and .sha256
# Requires: Xcode CLT, cmake >= 3.21, ninja; ggml submodule checked out.

OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$SRC/build-context-assist"
MACOS_MIN="14.0" # tauri.conf.json → bundle.macOS.minimumSystemVersion

REV="$(git -C "$SRC" rev-parse --short=12 HEAD)"
GGML_REV="$(git -C "$SRC/ggml" rev-parse --short=12 HEAD)"
TAG="${CRISPASR_DIST_TAG:-$REV}"
NAME="crispasr-$TAG-macos-arm64"

# Reproducibility: no build paths in objects, zero ar timestamps.
export ZERO_AR_DATE=1
PREFIX_MAP="-ffile-prefix-map=$SRC=. -ffile-prefix-map=$BUILD=build"

rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_MIN" \
  -DCMAKE_C_FLAGS="$PREFIX_MAP" \
  -DCMAKE_CXX_FLAGS="$PREFIX_MAP" \
  -DCMAKE_OBJC_FLAGS="$PREFIX_MAP" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CCACHE=OFF \
  -DCRISPASR_BUILD_TESTS=OFF \
  -DCRISPASR_BUILD_EXAMPLES=OFF \
  -DCRISPASR_BUILD_SERVER=OFF \
  -DCRISPASR_OPUS=OFF \
  -DCRISPASR_AMR=OFF \
  -DCRISPASR_NO_C2PA_NATIVE=ON
cmake --build "$BUILD" --target crispasr-lib --parallel

STAGE="$OUT/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib" "$STAGE/include"
# Sorted input → stable member order; -D → deterministic archive.
find "$BUILD" -name '*.a' | LC_ALL=C sort > "$BUILD/archives.txt"
libtool -static -D -no_warning_for_no_symbols -o "$STAGE/lib/libcrispasr.a" $(cat "$BUILD/archives.txt")
cp "$SRC/include/crispasr_session.h" "$SRC/include/crispasr.h" "$STAGE/include/"
cp "$SRC/LICENSE" "$STAGE/LICENSE"
cp "$SRC/ggml/LICENSE" "$STAGE/LICENSE.ggml"
cp "$SRC/THIRD_PARTY_NOTICES.txt" "$STAGE/"

LIB_SHA="$(shasum -a 256 "$STAGE/lib/libcrispasr.a" | cut -d' ' -f1)"
cat > "$STAGE/BUILD-INFO" <<EOF
crispasr_rev=$(git -C "$SRC" rev-parse HEAD)
ggml_rev=$(git -C "$SRC/ggml" rev-parse HEAD)
version=$(cat "$SRC/VERSION")
arch=arm64
macos_min=$MACOS_MIN
cmake=$(cmake --version | head -1)
clang=$(clang --version | head -1)
sdk=$(xcrun --show-sdk-version)
libcrispasr_sha256=$LIB_SHA
link=static=crispasr dylib=c++ framework=Foundation framework=Metal framework=Accelerate
EOF

# Deterministic tarball: fixed mtimes and owners, sorted entries, gzip without name/time.
find "$STAGE" -exec touch -h -t 200001010000 {} +
(cd "$OUT" && find "$NAME" | LC_ALL=C sort > "$BUILD/entries.txt" \
  && tar --uid 0 --gid 0 --uname '' --gname '' -n -cf - -T "$BUILD/entries.txt" | gzip -n -9 > "$NAME.tar.gz")
(cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "crispasr $REV (ggml $GGML_REV) -> $OUT/$NAME.tar.gz"
cat "$OUT/$NAME.tar.gz.sha256"
echo "libcrispasr.a sha256 $LIB_SHA"
