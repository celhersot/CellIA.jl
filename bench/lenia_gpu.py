# =============================================================================
#  Benchmark Lenia — GPU (CuPy + cuFFT)  ->  referencia "con GPU"
# =============================================================================
# Mismo algoritmo que las versiones CPU, pero TODO el bucle corre en la GPU
# (NVIDIA, via CuPy/cuFFT). La semilla se construye en CPU y se copia a la GPU
# UNA vez; no hay transferencias host<->device dentro de la region cronometrada.
# El cronometro se sincroniza con la GPU (deviceSynchronize) antes y despues del
# bucle, porque las operaciones CUDA son asincronas.
#
# Precision: Float64 (igual que CPU, para que la comparacion sea justa). En GPUs
# de consumo el FP64 esta limitado (~1/32), asi que es esperable que una GPU
# debil (p.ej. MX130) NO supere a la CPU en rejillas pequeñas.
#
# Uso:  python bench/lenia_gpu.py
#       BENCH_QUICK=1 python bench/lenia_gpu.py
# =============================================================================
import os, time
import numpy as np
import cupy as cp

MU, SIGMA, DT, R = 0.15, 0.017, 0.1, 13
PITCH = 64
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "results.csv"))


def read_seed(path):
    with open(path) as f:
        ph, pw = map(int, f.readline().split())
        seed = np.array([[float(v) for v in f.readline().split()] for _ in range(ph)],
                        dtype=np.float64)
    return seed


def build_kernel_host(dims):
    idx = np.arange(dims)
    d = np.minimum(idx, dims - idx)
    r = np.sqrt(d[:, None].astype(np.float64) ** 2 + d[None, :].astype(np.float64) ** 2) / R
    K = np.zeros_like(r)
    m = r < 1.0
    K[m] = np.exp(4.0 - 1.0 / (r[m] * (1.0 - r[m]) + 1e-10))
    s = K.sum()
    if s > 0:
        K /= s
    return K


def stamp_host(dims, seed):
    A = np.zeros((dims, dims), dtype=np.float64)
    ph, pw = seed.shape
    if dims < PITCH:
        ox = (dims - ph) // 2; oy = (dims - pw) // 2
        A[ox:ox + ph, oy:oy + pw] = seed
        return A
    nt = dims // PITCH
    io = (PITCH - ph) // 2; jo = (PITCH - pw) // 2
    for ti in range(nt):
        for tj in range(nt):
            ox = ti * PITCH + io; oy = tj * PITCH + jo
            A[ox:ox + ph, oy:oy + pw] = seed
    return A


def run_grid(dims, seed, steps, warmup, reps):
    Kfft = cp.fft.fft2(cp.asarray(build_kernel_host(dims)))  # kernel FFT en GPU (una vez)
    A0_host = stamp_host(dims, seed)

    def step(A):
        U = cp.real(cp.fft.ifft2(cp.fft.fft2(A) * Kfft))
        G = 2.0 * cp.exp(-((U - MU) ** 2) / (2.0 * SIGMA ** 2)) - 1.0
        cp.clip(A + DT * G, 0.0, 1.0, out=A)
        return A

    # warm-up: crea/cachea el plan cuFFT y compila (NVRTC) los kernels elementwise
    A = cp.asarray(A0_host)
    for _ in range(warmup):
        step(A)
    cp.cuda.Stream.null.synchronize()

    ms = []
    Efin = mfin = 0.0
    for _ in range(reps):
        A = cp.asarray(A0_host)              # estado inicial identico (copia host->device, fuera del cronometro)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        # ---------- REGION CRONOMETRADA (solo GPU) ----------
        for _ in range(steps):
            step(A)
        cp.cuda.Stream.null.synchronize()    # esperar a que la GPU termine de verdad
        # ----------------------------------------------------
        dt_s = time.perf_counter() - t0
        ms.append(1000.0 * dt_s / steps)
        Efin = float(A.sum()); mfin = float(A.max())
    return ms, Efin, mfin


def main():
    quick = os.environ.get("BENCH_QUICK", "0") == "1"
    grids = ([int(x) for x in os.environ["BENCH_GRIDS"].split(",")]
             if "BENCH_GRIDS" in os.environ else ([128] if quick else [128, 256, 512]))
    steps = int(os.environ.get("BENCH_STEPS", 50 if quick else 1000))
    warmup = int(os.environ.get("BENCH_WARMUP", 10 if quick else 50))
    reps = int(os.environ.get("BENCH_REPS", 2 if quick else 10))

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode()
    cc = cp.cuda.Device(0).compute_capability
    seed = read_seed(os.path.join(HERE, "orbium_seed.txt"))
    print(f"== GPU CuPy/cuFFT | {name} (CC {cc}) | FP64 | steps={steps} reps={reps} ==")

    with open(RESULTS, "a") as io:
        for dims in grids:
            ms, Efin, mfin = run_grid(dims, seed, steps, warmup, reps)
            med = float(np.median(ms))
            sd = float(np.std(ms)) if len(ms) > 1 else 0.0
            total = med * steps / 1000
            print(f"  {dims:4d}x{dims:<4d}  {med:8.3f} ms/step (+/-{sd:.3f})  "
                  f"total={total:.3f} s  E={Efin:.6f}  max={mfin:.6f}")
            for rep, v in enumerate(ms, 1):
                io.write(f"gpu_cupy,cuFFT,GPU,{dims},{rep},{v:.6f},"
                         f"{v*steps/1000:.6f},{Efin:.9f},{mfin:.9f}\n")


if __name__ == "__main__":
    main()
