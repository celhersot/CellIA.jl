# =============================================================================
#  Benchmark Lenia — PYTHON (NumPy / pyFFTW)  ->  baseline de la comunidad
# =============================================================================
# Replica EXACTA de la matematica de Cell_IA sobre un array crudo. NO renderiza
# nada. Solo se cronometra el bucle de pasos.
#
#   --backend numpy   : np.fft (pocketfft, el flujo habitual de un investigador)
#   --backend pyfftw  : FFTW via pyFFTW (mismo backend FFT que C/Julia)
#
# Uso:  python bench/lenia_py.py --backend numpy
#       BENCH_QUICK=1 python bench/lenia_py.py --backend pyfftw
# =============================================================================
import os, sys, time, argparse
import numpy as np

MU, SIGMA, DT, R = 0.15, 0.017, 0.1, 13
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "results.csv"))


def read_seed(path):
    with open(path) as f:
        ph, pw = map(int, f.readline().split())
        seed = np.array([[float(v) for v in f.readline().split()] for _ in range(ph)],
                        dtype=np.float64)
    assert seed.shape == (ph, pw)
    return seed


def kernel_fn(r):
    # exp(4 - 1/(r(1-r)+1e-10)) si r<1, 0 si no  (identico a CustomEvolutionRules)
    out = np.zeros_like(r)
    m = r < 1.0
    out[m] = np.exp(4.0 - 1.0 / (r[m] * (1.0 - r[m]) + 1e-10))
    return out


def build_kernel(dims):
    idx = np.arange(dims)
    d = np.minimum(idx, dims - idx)          # distancia toroidal desde el origen (esquina)
    dx = d[:, None]
    dy = d[None, :]
    r = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2) / R
    K = kernel_fn(r)
    s = K.sum()
    if s > 0:
        K /= s
    return K


PITCH = 64  # tesela un orbium por celda PITCHxPITCH (densidad constante, R fijo)


def stamp(dims, seed):
    A = np.zeros((dims, dims), dtype=np.float64)
    ph, pw = seed.shape
    if dims < PITCH:                       # rejilla pequeña: un orbium centrado
        ox = (dims - ph) // 2
        oy = (dims - pw) // 2
        A[ox:ox + ph, oy:oy + pw] = seed
        return A
    nt = dims // PITCH
    io = (PITCH - ph) // 2
    jo = (PITCH - pw) // 2
    for ti in range(nt):
        for tj in range(nt):
            ox = ti * PITCH + io
            oy = tj * PITCH + jo
            A[ox:ox + ph, oy:oy + pw] = seed
    return A


def make_stepper(backend, dims, Kfft):
    """Devuelve una funcion step(A)->A que hace un paso de Lenia in-place sobre A."""
    if backend == "numpy":
        def step(A):
            U = np.real(np.fft.ifft2(np.fft.fft2(A) * Kfft))
            G = 2.0 * np.exp(-((U - MU) ** 2) / (2.0 * SIGMA ** 2)) - 1.0
            np.clip(A + DT * G, 0.0, 1.0, out=A)
            return A
        return step

    elif backend == "pyfftw":
        import pyfftw
        a = pyfftw.empty_aligned((dims, dims), dtype="complex128")
        b = pyfftw.empty_aligned((dims, dims), dtype="complex128")
        fwd = pyfftw.FFTW(a, b, axes=(0, 1), direction="FFTW_FORWARD",
                          flags=("FFTW_MEASURE",), threads=1)
        inv = pyfftw.FFTW(b, a, axes=(0, 1), direction="FFTW_BACKWARD",
                          flags=("FFTW_MEASURE",), threads=1)  # normalise_idft=True por defecto

        def step(A):
            a[:] = A                 # copia A (real) al buffer complejo alineado
            fwd()                    # b = fft(a)
            b[:] = b * Kfft          # convolucion en frecuencia
            inv()                    # a = ifft(b) (normalizado)
            U = a.real
            G = 2.0 * np.exp(-((U - MU) ** 2) / (2.0 * SIGMA ** 2)) - 1.0
            np.clip(A + DT * G, 0.0, 1.0, out=A)
            return A
        return step
    else:
        raise ValueError(backend)


def run_grid(backend, dims, seed, steps, warmup, reps):
    K = build_kernel(dims)
    Kfft = np.fft.fft2(K)
    step = make_stepper(backend, dims, Kfft)

    # warm-up (plan FFTW / cache pocketfft / fallos de pagina)
    A = stamp(dims, seed)
    for _ in range(warmup):
        step(A)

    ms = []
    Efin = mfin = 0.0
    for _ in range(reps):
        A = stamp(dims, seed)        # estado inicial identico (fuera del cronometro)
        t0 = time.perf_counter()
        for _ in range(steps):
            step(A)
        dt_s = time.perf_counter() - t0
        ms.append(1000.0 * dt_s / steps)
        Efin = float(A.sum()); mfin = float(A.max())
    return ms, Efin, mfin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "pyfftw"], default="numpy")
    args = ap.parse_args()

    quick = os.environ.get("BENCH_QUICK", "0") == "1"
    grids = ([int(x) for x in os.environ["BENCH_GRIDS"].split(",")]
             if "BENCH_GRIDS" in os.environ else ([128] if quick else [128, 256, 512]))
    steps = int(os.environ.get("BENCH_STEPS", 50 if quick else 1000))
    warmup = int(os.environ.get("BENCH_WARMUP", 10 if quick else 50))
    reps = int(os.environ.get("BENCH_REPS", 2 if quick else 10))

    seed = read_seed(os.path.join(HERE, "orbium_seed.txt"))
    lang = "python_numpy" if args.backend == "numpy" else "python_pyfftw"
    bname = "pocketfft" if args.backend == "numpy" else "FFTW(pyfftw)"
    print(f"== Python {args.backend} | steps={steps} reps={reps} ==")
    print(f"   (OMP={os.environ.get('OMP_NUM_THREADS','?')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS','?')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS','?')})")

    with open(RESULTS, "a") as io:
        for dims in grids:
            ms, Efin, mfin = run_grid(args.backend, dims, seed, steps, warmup, reps)
            med = float(np.median(ms))
            sd = float(np.std(ms)) if len(ms) > 1 else 0.0
            total = med * steps / 1000
            print(f"  {dims:4d}x{dims:<4d}  {med:8.3f} ms/step (+/-{sd:.3f})  "
                  f"total={total:.3f} s  E={Efin:.6f}  max={mfin:.6f}")
            for rep, v in enumerate(ms, 1):
                io.write(f"{lang},{bname},CPU,{dims},{rep},{v:.6f},"
                         f"{v*steps/1000:.6f},{Efin:.9f},{mfin:.9f}\n")


if __name__ == "__main__":
    main()
