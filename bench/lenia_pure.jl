# =============================================================================
#  Benchmark Lenia — JULIA PURO (FFTW.jl), sin Agents.jl  ->  techo del lenguaje
# =============================================================================
# Replica EXACTA de la matematica de Cell_IA (CustomEvolutionRules.jl) sobre un
# array crudo, con planes FFTW preplaneados e in-place (Julia optimo).
# NO renderiza nada. Solo se cronometra el bucle de pasos.
#
# Uso:  julia -t1 bench/lenia_pure.jl
# Config rapida (smoke test):  BENCH_QUICK=1 julia -t1 bench/lenia_pure.jl
# =============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using FFTW
using Printf
using Statistics

FFTW.set_num_threads(1)   # linea base de 1 hilo (justo, sin ruido del planificador)

const MU    = 0.15
const SIGMA = 0.017
const DT    = 0.1
const R     = 13
const RESULTS = get(ENV, "BENCH_RESULTS", joinpath(@__DIR__, "results.csv"))

# --- semilla compartida (orbium 20x20) ---
function read_seed(path)
    lines = readlines(path)
    ph, pw = parse.(Int, split(lines[1]))
    seed = Matrix{Float64}(undef, ph, pw)
    for i in 1:ph
        seed[i, :] = parse.(Float64, split(lines[i+1]))
    end
    return seed
end

# kernel y crecimiento canonicos (identicos a CustomEvolutionRules)
kernel_fn(r) = r < 1.0 ? exp(4.0 - 1.0 / (r * (1.0 - r) + 1e-10)) : 0.0

function build_kernel_fft(dims::Int, R::Int)
    K = zeros(Float64, dims, dims)
    for x in 1:dims, y in 1:dims
        dx = min(x - 1, dims - (x - 1))
        dy = min(y - 1, dims - (y - 1))
        r  = sqrt(Float64(dx^2 + dy^2)) / R
        K[x, y] = kernel_fn(r)
    end
    s = sum(K); s > 0.0 && (K ./= s)
    return fft(K)
end

# Sembrado: tesela orbiums a densidad constante (uno por celda PITCH x PITCH).
# R fijo => kernel y dinamica identicos a toda escala; el campo se llena proporcional
# al area (4 orbiums en 128^2, 16 en 256^2, 64 en 512^2). El coste por paso depende
# solo del tamaño de rejilla, no del contenido (la FFT procesa toda la malla).
const PITCH = 64
function stamp!(A::Matrix{Float64}, seed::Matrix{Float64})
    dims = size(A, 1)
    ph, pw = size(seed)
    fill!(A, 0.0)
    if dims < PITCH                                   # rejilla pequeña: un orbium centrado
        ox = (dims - ph) ÷ 2; oy = (dims - pw) ÷ 2
        @inbounds A[ox+1:ox+ph, oy+1:oy+pw] .= seed
        return A
    end
    nt = dims ÷ PITCH
    io = (PITCH - ph) ÷ 2; jo = (PITCH - pw) ÷ 2
    for ti in 0:nt-1, tj in 0:nt-1
        ox = ti * PITCH + io; oy = tj * PITCH + jo
        @inbounds A[ox+1:ox+ph, oy+1:oy+pw] .= seed
    end
    return A
end

# un paso de Lenia (sincrono): U = real(ifft(fft(A).*Kfft)); A = clamp(A+dt*G(U),0,1)
function step!(A, Kfft, buf, P!, Pi!)
    @inbounds @. buf = complex(A)
    P! * buf
    @inbounds @. buf *= Kfft
    Pi! * buf
    @inbounds @. A = clamp(A + DT * (2.0 * exp(-((real(buf) - MU)^2) / (2.0 * SIGMA^2)) - 1.0), 0.0, 1.0)
    return A
end

function run_grid(dims::Int, seed, steps::Int, warmup::Int, reps::Int)
    Kfft = build_kernel_fft(dims, R)
    buf  = zeros(ComplexF64, dims, dims)
    P!   = plan_fft!(buf;  flags = FFTW.MEASURE)
    Pi!  = plan_ifft!(buf; flags = FFTW.MEASURE)
    A    = zeros(Float64, dims, dims)

    # warm-up (descarta JIT + planificacion FFTW + fallos de pagina)
    stamp!(A, seed)
    for _ in 1:warmup; step!(A, Kfft, buf, P!, Pi!); end

    ms = Float64[]
    Efin = 0.0; mfin = 0.0
    for _ in 1:reps
        stamp!(A, seed)                      # estado inicial identico (fuera del cronometro)
        t0 = time_ns()
        for _ in 1:steps; step!(A, Kfft, buf, P!, Pi!); end
        dt_s = (time_ns() - t0) / 1e9
        push!(ms, 1000.0 * dt_s / steps)
        Efin = sum(A); mfin = maximum(A)
    end
    return ms, Efin, mfin
end

function main()
    seed = read_seed(joinpath(@__DIR__, "orbium_seed.txt"))
    quick = get(ENV, "BENCH_QUICK", "0") == "1"
    grids  = haskey(ENV, "BENCH_GRIDS") ? parse.(Int, split(ENV["BENCH_GRIDS"], ",")) : (quick ? [128] : [128, 256, 512])
    steps  = parse(Int, get(ENV, "BENCH_STEPS",  string(quick ? 50 : 1000)))
    warmup = parse(Int, get(ENV, "BENCH_WARMUP", string(quick ? 10 : 50)))
    reps   = parse(Int, get(ENV, "BENCH_REPS",   string(quick ? 2 : 10)))

    println("== Julia puro (FFTW.jl) | hilos=$(FFTW.get_num_threads()) | steps=$steps reps=$reps ==")
    for dims in grids
        ms, Efin, mfin = run_grid(dims, seed, steps, warmup, reps)
        med = median(ms); sd = length(ms) > 1 ? std(ms) : 0.0
        total = med * steps / 1000
        @printf("  %4dx%-4d  %8.3f ms/step (+/-%.3f)  total=%.3f s  E=%.6f  max=%.6f\n",
                dims, dims, med, sd, total, Efin, mfin)
        open(RESULTS, "a") do io
            for (rep, v) in enumerate(ms)
                @printf(io, "julia_pure,FFTW,CPU,%d,%d,%.6f,%.6f,%.9f,%.9f\n",
                        dims, rep, v, v * steps / 1000, Efin, mfin)
            end
        end
    end
end

main()
