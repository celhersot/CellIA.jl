#!/usr/bin/env bash
# Compila bench/lenia.c contra el MISMO FFTW que usa Julia (el de FFTW_jll),
# con gcc de MSYS2 UCRT64. Reproducible: localiza el artifact de FFTW_jll solo.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GCC_BIN="/c/msys64/ucrt64/bin"
[ -x "$GCC_BIN/gcc.exe" ] || { echo "No encuentro gcc en $GCC_BIN (instala MSYS2 UCRT64)"; exit 1; }
export PATH="$GCC_BIN:$PATH"     # gcc necesita sus subprocesos (cc1/as/ld) en el PATH

# Localizar el artifact de FFTW_jll (cabecera fftw3.h + import lib libfftw3.dll.a + DLL)
DLL="$(find "$HOME/.julia/artifacts" -iname 'libfftw3-3.dll' 2>/dev/null | head -1)"
[ -n "$DLL" ] || { echo "No encuentro libfftw3-3.dll en ~/.julia/artifacts (instancia el proyecto Julia)"; exit 1; }
ART="$(dirname "$(dirname "$DLL")")"
echo "FFTW_jll artifact: $ART"

cp "$ART/bin/libfftw3-3.dll" "$ROOT/bench/"     # la DLL debe estar junto al exe en runtime

gcc -O3 -march=native -static-libgcc \
    -I "$ART/include" \
    -o "$ROOT/bench/lenia.exe" \
    "$ROOT/bench/lenia.c" \
    -L "$ART/lib" -lfftw3 -lm

echo "OK -> $ROOT/bench/lenia.exe"
