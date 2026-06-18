# =============================================================================
#  Profiling de Lenia en Cell_IA  —  ¿qué funciones consumen más tiempo?
# =============================================================================
# Usa el profiler estadístico de la biblioteca estándar de Julia (`Profile` +
# la macro `@profile`): muestrea la pila de llamadas mientras corre el bucle de
# pasos y produce un ranking de las funciones donde cae el programa más a menudo.
#
# Compara las DOS versiones del motor que conviven en el código:
#   - lenia_model_step!         -> motor OPTIMIZADO  (rama T06)
#   - lenia_model_step_legacy!  -> motor ORIGINAL     (rama T05)
#
# Igual que en lenia_cellia.jl, SOLO se perfila el bucle de simulación
# (step!), nunca el renderizado ni la E/S. El warm-up (JIT + planes FFTW) queda
# fuera del muestreo.
#
# Uso:
#   julia -t1 bench/profile_lenia.jl                          # optimizado, 256²
#   julia -t1 bench/profile_lenia.jl lenia_model_step_legacy! # legacy, 256²
#   julia -t1 bench/profile_lenia.jl lenia_model_step! 512    # optimizado, 512²
#
# Variables de entorno equivalentes:
#   BENCH_CELLIA_STEP   nombre de la función de paso (igual que el bench)
#   BENCH_DIM           tamaño de rejilla (lado)
#   PROFILE_SECONDS     segundos objetivo de muestreo (def. 6)
# =============================================================================
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TOML
include(joinpath(@__DIR__, "..", "src", "spaces", "HexagonalSpace.jl"))
include(joinpath(@__DIR__, "..", "src", "UniversalAgents.jl"))
include(joinpath(@__DIR__, "..", "src", "CustomEvolutionRules.jl"))
include(joinpath(@__DIR__, "..", "src", "SpaceDefinition.jl"))
include(joinpath(@__DIR__, "..", "src", "Initialization.jl"))
include(joinpath(@__DIR__, "..", "src", "Representation.jl"))

using .Initialization
using Agents
using FFTW
using Printf
using Profile

FFTW.set_num_threads(1)

# --- parámetros (CLI > env > defecto) --------------------------------------
STEP_FN = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "BENCH_CELLIA_STEP", "lenia_model_step!")
DIM     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : parse(Int, get(ENV, "BENCH_DIM", "256"))
TARGET_S = parse(Float64, get(ENV, "PROFILE_SECONDS", "6.0"))
LANG    = STEP_FN == "lenia_model_step_legacy!" ? "legacy_T05" : "optimizado_T06"
OUTDIR  = joinpath(@__DIR__, "profile")
mkpath(OUTDIR)

# --- mismo modelo/semilla que lenia_cellia.jl ------------------------------
const PITCH = 64

make_config(dims::Int) = Dict(
    "simulation" => Dict("model_name" => "prof_$(dims)", "seed" => 42),
    "space"      => Dict("type" => "grid", "dimensions" => [dims, dims],
                         "periodic" => true, "metric" => "chebyshev"),
    "agents"     => Dict("state_type" => "Float64"),
    "population" => Dict("pop_density" => Dict("0.0" => 1.0)),
    "properties" => Dict("lenia_mu" => 0.15, "lenia_sigma" => 0.017, "dt" => 0.1,
                         "kernel_radius" => 13, "kernel_type" => "gaussian"),
    "rules"      => Dict("initialization_rule" => "lenia_orbium",
                         "post_init" => "lenia_init!",
                         "model_step" => STEP_FN),
)

function read_seed(path)
    lines = readlines(path)
    ph, pw = parse.(Int, split(lines[1]))
    seed = Matrix{Float64}(undef, ph, pw)
    for i in 1:ph
        seed[i, :] = parse.(Float64, split(lines[i+1]))
    end
    return seed
end

function tiled_field(dims::Int, seed::Matrix{Float64})
    A = zeros(Float64, dims, dims)
    ph, pw = size(seed)
    if dims < PITCH
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

function apply_field!(model, A)
    for agent in allagents(model)
        agent.state = A[agent.pos[1], agent.pos[2]]
    end
    return model
end

function main()
    seed  = read_seed(joinpath(@__DIR__, "orbium_seed.txt"))
    A0    = tiled_field(DIM, seed)
    model = initialize_model(make_config(DIM))
    apply_field!(model, A0)

    println("== Profiling Cell_IA | step=$STEP_FN | $(DIM)x$(DIM) | hilos=$(FFTW.get_num_threads()) ==")

    # --- warm-up: JIT + planes FFTW (fuera del muestreo) -------------------
    step!(model, 20)

    # --- calibrar nº de pasos para ~TARGET_S segundos de muestreo ----------
    apply_field!(model, A0)
    t0 = time_ns(); step!(model, 10); ms_per_step = (time_ns() - t0) / 1e6 / 10
    nsteps = max(50, ceil(Int, TARGET_S * 1000 / ms_per_step))
    @printf("   ~%.3f ms/step  ->  perfilando %d pasos (~%.1f s)\n", ms_per_step, nsteps, nsteps * ms_per_step / 1000)

    # --- muestreo fino y profiling SOLO del bucle de pasos -----------------
    apply_field!(model, A0)
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.0005)      # ~2 kHz de muestreo
    @profile step!(model, nsteps)

    nsamples = length(Profile.fetch())
    @printf("   muestras recogidas: %d\n\n", nsamples)

    # --- reporte plano SIN frames de C (hotspots a nivel Julia) ------------
    flat_path = joinpath(OUTDIR, "flat_$(LANG)_$(DIM).txt")
    open(flat_path, "w") do io
        println(io, "Profiling Cell_IA — $STEP_FN — $(DIM)x$(DIM) — $nsteps pasos — $nsamples muestras")
        println(io, "Formato plano (C=false): solo frames de Julia. 'Count' = nº de muestras")
        println(io, "cuya pila contiene esa línea (inclusivo).\n")
        Profile.print(io; format = :flat, sortedby = :count, C = false)
    end

    # --- reporte plano CON frames de C (incluye FFTW y GC) -----------------
    flatc_path = joinpath(OUTDIR, "flatC_$(LANG)_$(DIM).txt")
    open(flatc_path, "w") do io
        println(io, "Profiling Cell_IA — $STEP_FN — $(DIM)x$(DIM) — $nsteps pasos — $nsamples muestras")
        println(io, "Formato plano (C=true): incluye frames de C -> aquí SÍ aparece el coste de")
        println(io, "FFTW (la convolución) y de la recolección de basura (gc).\n")
        Profile.print(io; format = :flat, sortedby = :count, C = true)
    end

    println("FLAT (sin frames de C — hotspots de Julia):")
    Profile.print(; format = :flat, sortedby = :count, C = false, mincount = max(1, nsamples ÷ 200))

    # --- reporte en árbol (dónde se anida el tiempo) -----------------------
    tree_path = joinpath(OUTDIR, "tree_$(LANG)_$(DIM).txt")
    open(tree_path, "w") do io
        println(io, "Profiling Cell_IA — $STEP_FN — $(DIM)x$(DIM) — árbol de llamadas\n")
        Profile.print(io; format = :tree, C = false, mincount = max(1, nsamples ÷ 100))
    end

    # --- flame graph HTML (opcional, si StatProfilerHTML está disponible) --
    html_note = "no generado (StatProfilerHTML no instalado)"
    try
        @eval using StatProfilerHTML
        htmldir = joinpath(OUTDIR, "html_$(LANG)_$(DIM)")
        Base.invokelatest(getfield(StatProfilerHTML, :statprofilehtml); path = htmldir)
        html_note = htmldir
    catch e
        @info "Flame graph HTML omitido" exception = (e, catch_backtrace())
    end

    println("\n--- salidas ---")
    println("   plano (Julia) : $flat_path")
    println("   plano (con C) : $flatc_path")
    println("   árbol         : $tree_path")
    println("   html          : $html_note")
end

main()
