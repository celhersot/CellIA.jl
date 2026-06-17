/* ===========================================================================
 *  Benchmark Lenia — C + FFTW  ->  referencia de bajo nivel
 * ===========================================================================
 *  Replica EXACTA de la matematica de Cell_IA sobre un array crudo, usando la
 *  MISMA libreria FFTW que Julia (libfftw3-3.dll de FFTW_jll). NO renderiza
 *  nada: solo se cronometra el bucle de pasos (QueryPerformanceCounter).
 *
 *  Compilar (MSVC):  ver bench/build_c.bat
 *  Ejecutar:         bench\lenia.exe            (full)
 *                    bench\lenia.exe quick      (smoke test)
 * =========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <fftw3.h>

#define MU    0.15
#define SIGMA 0.017
#define DT    0.1
#define R     13

static double now_seconds(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* --- semilla orbium 20x20 compartida --- */
static int read_seed(const char *path, double **seed, int *ph, int *pw) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "No puedo abrir %s\n", path); return 0; }
    if (fscanf(f, "%d %d", ph, pw) != 2) { fclose(f); return 0; }
    int n = (*ph) * (*pw);
    *seed = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%lf", &(*seed)[i]) != 1) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

static double kernel_fn(double r) {
    return (r < 1.0) ? exp(4.0 - 1.0 / (r * (1.0 - r) + 1e-10)) : 0.0;
}

/* construye K (row-major), normaliza a suma 1, devuelve en out (real) */
static void build_kernel(double *K, int dims) {
    double s = 0.0;
    for (int x = 0; x < dims; ++x) {
        int dx = (x < dims - x) ? x : dims - x;        /* distancia toroidal */
        for (int y = 0; y < dims; ++y) {
            int dy = (y < dims - y) ? y : dims - y;
            double r = sqrt((double)(dx * dx + dy * dy)) / (double)R;
            double v = kernel_fn(r);
            K[x * dims + y] = v;
            s += v;
        }
    }
    if (s > 0.0) for (int i = 0; i < dims * dims; ++i) K[i] /= s;
}

/* Sembrado: tesela un orbium por celda PITCHxPITCH (densidad constante, R fijo) */
#define PITCH 64
static void stamp(double *A, int dims, const double *seed, int ph, int pw) {
    memset(A, 0, sizeof(double) * dims * dims);
    if (dims < PITCH) {                         /* rejilla pequeña: un orbium centrado */
        int ox = (dims - ph) / 2, oy = (dims - pw) / 2;
        for (int i = 0; i < ph; ++i)
            for (int j = 0; j < pw; ++j)
                A[(ox + i) * dims + (oy + j)] = seed[i * pw + j];
        return;
    }
    int nt = dims / PITCH;
    int io = (PITCH - ph) / 2, jo = (PITCH - pw) / 2;
    for (int ti = 0; ti < nt; ++ti)
        for (int tj = 0; tj < nt; ++tj) {
            int ox = ti * PITCH + io, oy = tj * PITCH + jo;
            for (int i = 0; i < ph; ++i)
                for (int j = 0; j < pw; ++j)
                    A[(ox + i) * dims + (oy + j)] = seed[i * pw + j];
        }
}

/* compara para mediana */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static void run_grid(int dims, const double *seed, int ph, int pw,
                     int steps, int warmup, int reps, FILE *csv) {
    int N = dims * dims;
    double invN = 1.0 / (double)N;

    double *A    = (double *)malloc(sizeof(double) * N);
    double *Kr   = (double *)malloc(sizeof(double) * N);
    fftw_complex *buf  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *Kfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    /* planes FFTW in-place sobre buf (mismo rigor que Julia/pyfftw: MEASURE) */
    fftw_plan fwd = fftw_plan_dft_2d(dims, dims, buf, buf, FFTW_FORWARD,  FFTW_MEASURE);
    fftw_plan bwd = fftw_plan_dft_2d(dims, dims, buf, buf, FFTW_BACKWARD, FFTW_MEASURE);

    /* FFT del kernel (una vez) */
    build_kernel(Kr, dims);
    for (int i = 0; i < N; ++i) { buf[i][0] = Kr[i]; buf[i][1] = 0.0; }
    fftw_execute(fwd);
    memcpy(Kfft, buf, sizeof(fftw_complex) * N);

    /* warm-up (no cronometrado) */
    stamp(A, dims, seed, ph, pw);
    for (int t = 0; t < warmup; ++t) {
        for (int i = 0; i < N; ++i) { buf[i][0] = A[i]; buf[i][1] = 0.0; }
        fftw_execute(fwd);
        for (int i = 0; i < N; ++i) {
            double re = buf[i][0] * Kfft[i][0] - buf[i][1] * Kfft[i][1];
            double im = buf[i][0] * Kfft[i][1] + buf[i][1] * Kfft[i][0];
            buf[i][0] = re; buf[i][1] = im;
        }
        fftw_execute(bwd);
        for (int i = 0; i < N; ++i) {
            double u = buf[i][0] * invN;
            double g = 2.0 * exp(-((u - MU) * (u - MU)) / (2.0 * SIGMA * SIGMA)) - 1.0;
            double v = A[i] + DT * g;
            A[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
        }
    }

    double *ms = (double *)malloc(sizeof(double) * reps);
    double Efin = 0.0, mfin = 0.0;
    for (int rep = 0; rep < reps; ++rep) {
        stamp(A, dims, seed, ph, pw);          /* estado inicial identico (fuera del cronometro) */
        double t0 = now_seconds();
        /* ---------- REGION CRONOMETRADA ---------- */
        for (int t = 0; t < steps; ++t) {
            for (int i = 0; i < N; ++i) { buf[i][0] = A[i]; buf[i][1] = 0.0; }
            fftw_execute(fwd);
            for (int i = 0; i < N; ++i) {
                double re = buf[i][0] * Kfft[i][0] - buf[i][1] * Kfft[i][1];
                double im = buf[i][0] * Kfft[i][1] + buf[i][1] * Kfft[i][0];
                buf[i][0] = re; buf[i][1] = im;
            }
            fftw_execute(bwd);
            for (int i = 0; i < N; ++i) {
                double u = buf[i][0] * invN;
                double g = 2.0 * exp(-((u - MU) * (u - MU)) / (2.0 * SIGMA * SIGMA)) - 1.0;
                double v = A[i] + DT * g;
                A[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
            }
        }
        /* ----------------------------------------- */
        double dt_s = now_seconds() - t0;
        ms[rep] = 1000.0 * dt_s / (double)steps;

        Efin = 0.0; mfin = 0.0;
        for (int i = 0; i < N; ++i) { Efin += A[i]; if (A[i] > mfin) mfin = A[i]; }
    }

    /* mediana + desv */
    double *sorted = (double *)malloc(sizeof(double) * reps);
    memcpy(sorted, ms, sizeof(double) * reps);
    qsort(sorted, reps, sizeof(double), cmp_double);
    double med = (reps % 2) ? sorted[reps/2] : 0.5 * (sorted[reps/2-1] + sorted[reps/2]);
    double mean = 0.0; for (int i=0;i<reps;++i) mean += ms[i]; mean /= reps;
    double sd = 0.0; for (int i=0;i<reps;++i) sd += (ms[i]-mean)*(ms[i]-mean);
    sd = reps>1 ? sqrt(sd/(reps-1)) : 0.0;

    printf("  %4dx%-4d  %8.3f ms/step (+/-%.3f)  total=%.3f s  E=%.6f  max=%.6f\n",
           dims, dims, med, sd, med*steps/1000.0, Efin, mfin);

    for (int rep = 0; rep < reps; ++rep)
        fprintf(csv, "c,FFTW,CPU,%d,%d,%.6f,%.6f,%.9f,%.9f\n",
                dims, rep+1, ms[rep], ms[rep]*steps/1000.0, Efin, mfin);

    free(sorted); free(ms);
    fftw_destroy_plan(fwd); fftw_destroy_plan(bwd);
    fftw_free(buf); fftw_free(Kfft); free(A); free(Kr);
}

int main(int argc, char **argv) {
    int quick = (argc > 1 && strcmp(argv[1], "quick") == 0);
    int grids_buf[16];
    int grids_full[3] = {128, 256, 512};
    int grids_quick[1] = {128};
    int *grids = quick ? grids_quick : grids_full;
    int ngrids = quick ? 1 : 3;
    int steps  = quick ? 50 : 1000;
    int warmup = quick ? 10 : 50;
    int reps   = quick ? 2  : 10;

    /* overrides por entorno (el driver controla el coste) */
    const char *e;
    if ((e = getenv("BENCH_STEPS")))  steps  = atoi(e);
    if ((e = getenv("BENCH_WARMUP"))) warmup = atoi(e);
    if ((e = getenv("BENCH_REPS")))   reps   = atoi(e);
    if ((e = getenv("BENCH_GRIDS"))) {
        ngrids = 0;
        char tmp[128]; snprintf(tmp, sizeof(tmp), "%s", e);
        for (char *tok = strtok(tmp, ","); tok && ngrids < 16; tok = strtok(NULL, ","))
            grids_buf[ngrids++] = atoi(tok);
        grids = grids_buf;
    }

    const char *here = "bench/";
    char seedpath[512], csvpath[512];
    snprintf(seedpath, sizeof(seedpath), "%sorbium_seed.txt", here);
    snprintf(csvpath,  sizeof(csvpath),  "%sresults.csv", here);

    double *seed; int ph, pw;
    if (!read_seed(seedpath, &seed, &ph, &pw)) {
        /* intento alternativo: ejecutado desde dentro de bench/ */
        if (!read_seed("orbium_seed.txt", &seed, &ph, &pw)) return 1;
        snprintf(csvpath, sizeof(csvpath), "results.csv");
    }

    const char *env_csv = getenv("BENCH_RESULTS");   /* el driver puede redirigir el CSV */
    if (env_csv && env_csv[0]) snprintf(csvpath, sizeof(csvpath), "%s", env_csv);

    FILE *csv = fopen(csvpath, "a");
    if (!csv) { fprintf(stderr, "No puedo abrir %s\n", csvpath); return 1; }

    printf("== C (FFTW) | 1 hilo | steps=%d reps=%d ==\n", steps, reps);
    for (int g = 0; g < ngrids; ++g)
        run_grid(grids[g], seed, ph, pw, steps, warmup, reps, csv);

    fclose(csv);
    free(seed);
    return 0;
}
