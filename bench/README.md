# Benchmark de rendimiento de Lenia (C · Python · Julia · Cell_IA · GPU)

Estudio de rendimiento **puro** del motor de Lenia: convolución circular por FFT +
función de crecimiento + clamp, **sin renderizado ni I/O dentro del cronómetro**.
Iteración 5 del TFM (rama `T05-exp-rendimiento`). Plan completo en [`../specs.md`](../specs.md).

## Qué se compara

| # | Implementación | Backend FFT | Dispositivo | Fichero |
|---|---|---|---|---|
| 1 | **C** | FFTW (libfftw3) | CPU | [`lenia.c`](lenia.c) |
| 2 | **Julia puro** | FFTW.jl | CPU | [`lenia_pure.jl`](lenia_pure.jl) |
| 3 | **Cell_IA** (framework, Agents.jl) | FFTW.jl | CPU | [`lenia_cellia.jl`](lenia_cellia.jl) |
| 4 | **Python NumPy** | pocketfft (`np.fft`) | CPU | [`lenia_py.py --backend numpy`](lenia_py.py) |
| 5 | **Python pyFFTW** | FFTW (pyFFTW) | CPU | [`lenia_py.py --backend pyfftw`](lenia_py.py) |
| 6 | **Python CuPy** | cuFFT | **GPU** | [`lenia_gpu.py`](lenia_gpu.py) |

> **C y Julia usan el MISMO binario de FFTW** (el `libfftw3-3.dll` que distribuye
> `FFTW_jll`, el que ya usa Cell_IA). Así, la diferencia C ↔ Julia aísla el overhead
> de lenguaje/runtime con numéricos idénticos. **pyFFTW** usa también FFTW desde Python:
> separa el *backend FFT* del *overhead del intérprete* frente a NumPy/pocketfft.
> Las **4 filas de "Julia el lenguaje" vs "Cell_IA el framework"** muestran por separado
> el techo del lenguaje y el coste real de la abstracción de agentes (Agents.jl).

## El algoritmo (idéntico en las 6 implementaciones)

Réplica exacta del motor de Cell_IA ([`../src/CustomEvolutionRules.jl`](../src/CustomEvolutionRules.jl)):

```
A ∈ [0,1] (Float64), malla dims×dims, frontera periódica (toroidal)
Kernel K (precomputado una vez, se guarda su FFT):
    r = sqrt(dx²+dy²)/R con (dx,dy) = distancia toroidal desde la esquina
    K = (r<1) ? exp(4 − 1/(r(1−r)+1e-10)) : 0 ;   K ← K/ΣK ;   K_fft = fft(K)
Paso:  U = real(ifft(fft(A) · K_fft));  G(u)=2·exp(−(u−μ)²/2σ²)−1;  A = clamp(A+dt·G(U),0,1)
```

Parámetros (orbium canónico): `μ=0.15, σ=0.017, dt=0.1, R=13`, `Float64` en todos.
**Sin RNG ⇒ totalmente determinista y reproducible.**

### Semilla: orbiums teselados a densidad constante

El "organismo" es el **orbium** (Bert Chan), exportado del propio framework a
[`orbium_seed.txt`](orbium_seed.txt) (fuente única → semilla idéntica en todos los lenguajes).
Se **tesela un orbium por celda de 64×64** (4 en 128², 16 en 256², 64 en 512²), manteniendo
`R=13` fijo. Decisión razonada:

- El coste por paso depende **solo del tamaño de la malla** (la FFT procesa toda la rejilla
  y el `growth`/`clamp` recorre todas las celdas, haya 1 orbium o 64; el fondo a cero no se
  "salta"). **Teselar no cambia el ms/step medido**; sirve para que el campo esté poblado de
  forma realista a toda escala (fotos representativas) sin alterar la dinámica.
- Reescalar un único orbium exigiría cambiar `R` y reestabilizar el patrón (afinado a `R=13`):
  cambiaría kernel y dinámica → peor para comparar. Por eso se tesela en lugar de reescalar.

## Metodología de medición (justa y reproducible)

- **Solo se cronometra el bucle de pasos.** Nada de fotos/vídeo/CSV dentro del cronómetro.
  En Cell_IA, las fotos del estado inicial/final ([`photos/`](photos)) se guardan **fuera**
  del cronómetro; en los demás lenguajes no hay nada visual.
- **1 hilo** como línea base: `FFTW.set_num_threads(1)` + `julia -t1`; `OMP/MKL/OPENBLAS_NUM_THREADS=1`
  en Python; FFTW de C sin hilos; pyFFTW `threads=1`.
- **Warm-up** antes de medir (descarta JIT de Julia, creación del *plan* FFTW, fallos de página).
- **Planes FFTW preplaneados** (`FFTW_MEASURE`) en C, Julia puro y pyFFTW; kernel FFT precomputado
  una vez. NumPy usa pocketfft (cachea twiddles). En GPU el cronómetro **sincroniza** con la GPU
  (`deviceSynchronize`) antes y después del bucle (las operaciones CUDA son asíncronas).
- **Reloj monótono**: `QueryPerformanceCounter` (C), `time.perf_counter()` (Python), `time_ns()` (Julia).
- Métrica: **mediana de ms/step** sobre N repeticiones (la varianza es ~0 por ser determinista).
- **Verificación de correctitud cruzada**: tras la misma cantidad de pasos, las 6 implementaciones
  imprimen `E = ΣA`. A horizonte corto (50 pasos) coinciden a ~1e-9 (mismo algoritmo). A 1000 pasos
  puede haber ligera deriva entre backends FFT distintos (FFTW vs pocketfft vs cuFFT) por la
  no-asociatividad en coma flotante; es esperable y no afecta a la medida de tiempos.

## Verificación de uso de GPU

`run_all.py` monitoriza `nvidia-smi` durante **cada** simulación y decide si la GPU se usa a
partir de señales fiables: **proceso de cómputo en la GPU** y **incremento de VRAM** (la
utilización % de esta MX130 es ruidosa y solo se reporta como información). Resultado esperado:
las 5 implementaciones CPU **no** usan la GPU (incl. **Cell_IA**, porque FFTW.jl es solo-CPU);
solo CuPy aparece como proceso en la GPU con VRAM reservada.

## Entorno (esta máquina)

- **CPU**: Intel Core i7-8565U (Whiskey Lake, 4C/8T, AVX2). **RAM**: portátil.
- **GPU**: NVIDIA GeForce MX130 (Maxwell GM108, *compute capability* 5.0, 2 GB), driver 582.28.
- **SO**: Windows 11.
- **Julia** 1.11.7 + FFTW.jl (FFTW 3.3.10 vía `FFTW_jll`) + Agents.jl 6.2.10.
- **Python** 3.13.7 + NumPy 2.3.5 (OpenBLAS) + SciPy 1.16.3 + pyFFTW + CuPy 14.1.1 (runtime CUDA 12.9).
- **C**: gcc 15.2.0 (MSYS2 UCRT64), `-O3 -march=native -static-libgcc`, enlazado contra el FFTW de `FFTW_jll`.
- **CUDA**: toolkit 13.1 en el sistema; CuPy usa el runtime CUDA 12 (wheels `nvidia-*-cu12`).

> Nota GPU: la MX130 (CC 5.0) es muy modesta y su **FP64 está limitado** (~1/32). `curand` de
> CuPy 14 no trae kernels para sm_50, pero el benchmark **no usa RNG** (semilla determinista),
> así que solo dependemos de **cuFFT** (funciona en FP64) y kernels elementwise (NVRTC).

## Cómo reproducir

```bash
# 0. (una vez) exportar la semilla orbium desde el framework
julia bench/export_seed.jl

# 1. (una vez) compilar la versión C contra el FFTW de FFTW_jll
#    (gcc de MSYS2 UCRT64; ver bench/build_c.sh)
bash bench/build_c.sh

# 2. ejecutar todo (correctitud + tiempos + tabla + verificación GPU)
python bench/run_all.py                 # completo (1000 pasos, 5 reps)
python bench/run_all.py --quick         # smoke test (128², 50 pasos)
python bench/run_all.py --table-only    # regenerar la tabla desde results.csv
```

Salidas: [`results.csv`](results.csv) (datos crudos), [`results_table.md`](results_table.md)
(tabla agregada), [`photos/`](photos) (estado inicial/final de Cell_IA), [`run_full.log`](run_full.log)
(traza completa con los veredictos de GPU).
