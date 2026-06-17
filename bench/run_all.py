#!/usr/bin/env python
# =============================================================================
#  Driver del benchmark Lenia (C / Python / Julia / Cell_IA / GPU)
# =============================================================================
#  - Fija 1 hilo en las implementaciones CPU (justo, sin ruido del planificador).
#  - Ejecuta una PASADA DE CORRECTITUD corta (50 pasos) y comprueba que las 6
#    implementaciones coinciden en la energia E (mismo algoritmo, sin RNG).
#  - Ejecuta la PASADA DE TIEMPOS (1000 pasos) monitorizando nvidia-smi en cada
#    simulacion para VERIFICAR si la GPU se usa o no (peak util / proceso / VRAM).
#  - Agrega results.csv -> tabla Markdown + LaTeX en bench/.
#
#  Uso:  python bench/run_all.py            (completo)
#        python bench/run_all.py --quick    (rapido, smoke)
#        python bench/run_all.py --table-only
# =============================================================================
import os, sys, re, time, subprocess, threading, statistics, argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results.csv")
CHECK_CSV = os.path.join(HERE, "_check_discard.csv")
HEADER = "lang,backend,device,grid,rep,ms_per_step,total_s,E_final,max_final\n"

PY = sys.executable
EXE = os.path.join(HERE, "lenia.exe")

CPU_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

IMPLS = [
    dict(key="c",             label="C (FFTW)",                    device="CPU",
         cmd=[EXE],                                            env=CPU_ENV),
    dict(key="julia_pure",    label="Julia puro (FFTW.jl)",        device="CPU",
         cmd=["julia", "-t1", os.path.join(HERE, "lenia_pure.jl")], env=CPU_ENV),
    dict(key="cell_ia",       label="Cell_IA (Agents.jl + FFTW)",  device="CPU",
         cmd=["julia", "-t1", os.path.join(HERE, "lenia_cellia.jl")], env=CPU_ENV),
    dict(key="python_numpy",  label="Python NumPy (pocketfft)",    device="CPU",
         cmd=[PY, os.path.join(HERE, "lenia_py.py"), "--backend", "numpy"],  env=CPU_ENV),
    dict(key="python_pyfftw", label="Python pyFFTW (FFTW)",        device="CPU",
         cmd=[PY, os.path.join(HERE, "lenia_py.py"), "--backend", "pyfftw"], env=CPU_ENV),
    dict(key="gpu_cupy",      label="Python CuPy (cuFFT) [GPU]",   device="GPU",
         cmd=[PY, os.path.join(HERE, "lenia_gpu.py")],         env={}),
]
ORDER = [d["key"] for d in IMPLS]
LABEL = {d["key"]: d["label"] for d in IMPLS}
DEVICE = {d["key"]: d["device"] for d in IMPLS}

# ---------- monitorizacion de GPU (nvidia-smi) -------------------------------
def _smi(query, extra):
    try:
        out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"] + extra,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""

def gpu_available():
    return _smi("--query-gpu=name", []) != ""

def sample_util_mem():
    s = _smi("--query-gpu=utilization.gpu,memory.used", [])
    try:
        u, m = s.split(",")
        return int(u), int(m)
    except Exception:
        return None

def sample_procs():
    s = _smi("--query-compute-apps=process_name,used_memory", [])
    return [ln.strip() for ln in s.splitlines() if ln.strip()]

# ---------- ejecutar una implementacion con monitor GPU ----------------------
def run_impl(impl, env_extra, results_path, monitor):
    env = os.environ.copy()
    env.update(impl["env"])
    env.update(env_extra)
    env["BENCH_RESULTS"] = results_path

    stats = {"peak_util": 0, "peak_mem": 0, "base_mem": 0, "procs": set()}
    stop = threading.Event()
    if monitor and gpu_available():
        s0 = sample_util_mem()
        stats["base_mem"] = s0[1] if s0 else 0
        def mon():
            while not stop.is_set():
                s = sample_util_mem()
                if s:
                    stats["peak_util"] = max(stats["peak_util"], s[0])
                    stats["peak_mem"] = max(stats["peak_mem"], s[1])
                for p in sample_procs():
                    stats["procs"].add(p)
                time.sleep(0.1)
        th = threading.Thread(target=mon, daemon=True); th.start()
    else:
        th = None

    t0 = time.time()
    proc = subprocess.run(impl["cmd"], env=env, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    if th:
        stop.set(); th.join(timeout=2)
    return proc, stats, wall

def gpu_verdict(impl, stats):
    """Decide si la GPU se uso. Señales FIABLES: proceso de computo en la GPU y
    delta de VRAM. La utilizacion (%) es ruidosa en esta MX130 (sube sola por el
    SO/driver aunque no la usemos), asi que se reporta solo como informacion."""
    if not gpu_available():
        return "n/d (sin nvidia-smi)"
    names = set()
    for p in stats["procs"]:
        nm = os.path.basename(p.split(",")[0].strip())
        if any(k in nm.lower() for k in ("python", "julia", "lenia")):
            names.add(nm)
    mem_delta = stats["peak_mem"] - stats["base_mem"]
    # Señal DEFINITIVA: ¿hay un proceso de computo NUESTRO en la GPU? La VRAM total
    # puede derivar por el SO/driver/Makie en runs largos (falso positivo), asi que
    # solo cuenta como corroboracion cuando ya hay proceso de computo.
    used = len(names) > 0
    tag = "SI USA GPU" if used else "NO usa GPU (solo CPU)"
    return (f"{tag} [proc_GPU={sorted(names) or '-'} VRAM+={mem_delta}MiB(info) "
            f"util_pico={stats['peak_util']}%(info, ruidoso)]")

# ---------- pasada de correctitud (energia E debe coincidir) -----------------
def correctness_pass(grids="128", steps=50):
    print("\n================ PASADA DE CORRECTITUD (E debe coincidir) ================")
    env_extra = {"BENCH_GRIDS": str(grids), "BENCH_STEPS": str(steps),
                 "BENCH_REPS": "1", "BENCH_WARMUP": "5"}
    if os.path.exists(CHECK_CSV):
        os.remove(CHECK_CSV)
    energies = {}
    for impl in IMPLS:
        proc, _, _ = run_impl(impl, env_extra, CHECK_CSV, monitor=False)
        m = re.findall(r"E=([0-9.]+)", proc.stdout)
        e = float(m[-1]) if m else None
        energies[impl["key"]] = e
        status = f"E={e}" if e is not None else f"FALLO\n{proc.stdout[-400:]}\n{proc.stderr[-400:]}"
        print(f"  {LABEL[impl['key']]:32s} {status}")
    vals = [e for e in energies.values() if e is not None]
    ok = False
    if len(vals) == len(IMPLS):
        spread = (max(vals) - min(vals)) / max(abs(max(vals)), 1e-12)
        ok = spread < 1e-5
        print(f"  -> spread relativo = {spread:.2e}  =>  {'COINCIDEN [OK]' if ok else 'DIFIEREN [FAIL]'}")
    else:
        print("  -> alguna implementacion fallo; revisa arriba")
    return ok

# ---------- pasada de tiempos ------------------------------------------------
def timing_pass(grids, steps, reps, warmup):
    print("\n================ PASADA DE TIEMPOS (1 hilo CPU) ================")
    with open(RESULTS, "w") as f:
        f.write(HEADER)
    env_extra = {"BENCH_GRIDS": grids, "BENCH_STEPS": str(steps),
                 "BENCH_REPS": str(reps), "BENCH_WARMUP": str(warmup)}
    for impl in IMPLS:
        print(f"\n>>> {LABEL[impl['key']]}")
        proc, stats, wall = run_impl(impl, env_extra, RESULTS, monitor=True)
        for ln in proc.stdout.splitlines():
            if "ms/step" in ln or "==" in ln:
                print("    " + ln.strip())
        if proc.returncode != 0:
            print(f"    [!] returncode={proc.returncode}\n{proc.stderr[-600:]}")
        print(f"    GPU: {gpu_verdict(impl, stats)}   (wall={wall:.1f}s)")

# ---------- tabla ------------------------------------------------------------
def make_table(steps):
    if not os.path.exists(RESULTS):
        print("No hay results.csv"); return
    rows = {}
    grids = set()
    with open(RESULTS) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 9:
                continue
            lang, grid = p[0], int(p[3])
            ms, E = float(p[5]), float(p[7])
            rows.setdefault((lang, grid), {"ms": [], "E": E})["ms"].append(ms)
            grids.add(grid)
    grids = sorted(grids)

    def med(lang, g):
        r = rows.get((lang, g))
        return statistics.median(r["ms"]) if r else None

    # tabla de texto
    lines = []
    lines.append("\n================ TABLA DE RENDIMIENTO (mediana ms/step) ================")
    head = f"{'Implementacion':32s} " + " ".join(f"{str(g)+'x'+str(g):>14s}" for g in grids)
    lines.append(head)
    lines.append("-" * len(head))
    for k in ORDER:
        cells = []
        for g in grids:
            v = med(k, g)
            cells.append(f"{v:>14.3f}" if v is not None else f"{'--':>14s}")
        lines.append(f"{LABEL[k]:32s} " + " ".join(cells))
    print("\n".join(lines))

    # markdown
    mdl = ["# Resultados del benchmark Lenia (mediana ms/step, 1 hilo CPU salvo GPU)\n"]
    mdl.append(f"Pasos por simulacion: {steps}. Semilla: orbiums teselados (densidad constante, R=13).\n")
    cols = "| Implementacion | Dispositivo | " + " | ".join(f"{g}x{g} ms/step | {g}x{g} total (s)" for g in grids) + " |"
    sep = "|" + "---|" * (2 + 2 * len(grids))
    mdl.append(cols); mdl.append(sep)
    for k in ORDER:
        cells = []
        for g in grids:
            v = med(k, g)
            if v is None:
                cells += ["--", "--"]
            else:
                cells += [f"{v:.3f}", f"{v*steps/1000:.2f}"]
        mdl.append(f"| {LABEL[k]} | {DEVICE[k]} | " + " | ".join(cells) + " |")
    mdl.append("\n## Energia final E (verificacion; debe coincidir entre backends FFTW; "
               "ligera deriva esperable en pocketfft/cuFFT a horizonte largo)\n")
    mdl.append("| Implementacion | " + " | ".join(f"E @ {g}x{g}" for g in grids) + " |")
    mdl.append("|" + "---|" * (1 + len(grids)))
    for k in ORDER:
        cells = []
        for g in grids:
            r = rows.get((k, g))
            cells.append(f"{r['E']:.6f}" if r else "--")
        mdl.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    out = os.path.join(HERE, "results_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(mdl) + "\n")
    print(f"\nTabla Markdown -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--grids", type=str, default=None)
    args = ap.parse_args()

    steps = args.steps or (50 if args.quick else 1000)
    reps = args.reps or (2 if args.quick else 5)
    warmup = 10 if args.quick else 50
    grids = args.grids or ("128" if args.quick else "128,256,512")

    if args.table_only:
        make_table(steps); return

    print(f"GPU detectada: {_smi('--query-gpu=name', []) or 'NO'}")
    if not args.no_check:
        correctness_pass(grids="128", steps=50)
    timing_pass(grids, steps, reps, warmup)
    make_table(steps)


if __name__ == "__main__":
    main()
