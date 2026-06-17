# =============================================================================
#  Benchmark Lenia — CELL_IA (framework real: Agents.jl + FFTW.jl)
# =============================================================================
# Usa el motor del framework TAL CUAL (lenia_model_step!), con la abstraccion de
# agentes de Agents.jl. Mide el coste honesto del framework.
#
# SOLO se cronometra step!(model, steps). Las fotos del estado inicial y final
# (para verificar visualmente que la simulacion ocurre) se guardan FUERA del
# cronometro, igual que pidio el usuario. Los demas lenguajes no renderizan nada.
#
# Uso:  julia -t1 bench/lenia_cellia.jl
#       BENCH_QUICK=1 julia -t1 bench/lenia_cellia.jl
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
using .Representation
using Agents
using FFTW
using Printf
using Statistics

FFTW.set_num_threads(1)

const RESULTS = get(ENV, "BENCH_RESULTS", joinpath(@__DIR__, "results.csv"))
const PHOTODIR = joinpath(@__DIR__, "photos")

# Config del modelo Cell_IA para una rejilla dada (mismo orbium/parametros que organismo.toml).
function make_config(dims::Int)
    Dict(
        "simulation" => Dict("model_name" => "bench_$(dims)", "seed" => 42),
        "space"      => Dict("type" => "grid", "dimensions" => [dims, dims],
                             "periodic" => true, "metric" => "chebyshev"),
        "agents"     => Dict("state_type" => "Float64"),
        "population" => Dict("pop_density" => Dict("0.0" => 1.0)),
        "properties" => Dict("lenia_mu" => 0.15, "lenia_sigma" => 0.017, "dt" => 0.1,
                             "kernel_radius" => 13, "kernel_type" => "gaussian"),
        "rules"      => Dict("initialization_rule" => "lenia_orbium",
                             "post_init" => "lenia_init!",
                             "model_step" => "lenia_model_step!"),
    )
end

# --- semilla compartida + teselado identico al de las demas implementaciones ---
const PITCH = 64
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

# Sobrescribe el estado de cada agente con el campo teselado (mismo inicio que las demas).
function apply_field!(model, A)
    for agent in allagents(model)
        agent.state = A[agent.pos[1], agent.pos[2]]
    end
    return model
end

function state_matrix(model)
    dims = size(abmspace(model))
    A = zeros(Float64, dims...)
    for agent in allagents(model)
        A[agent.pos[1], agent.pos[2]] = Float64(agent.state)
    end
    return A
end

energy(model) = sum(a.state for a in allagents(model))
maxstate(model) = maximum(a.state for a in allagents(model))

function run_grid(dims::Int, seed::Matrix{Float64}, steps::Int, warmup::Int, reps::Int; save_photos::Bool=true)
    A0 = tiled_field(dims, seed)
    model = initialize_model(make_config(dims))    # se construye UNA vez por rejilla (poblar es caro)

    # --- warm-up (JIT + wisdom FFTW), NO cronometrado ---
    apply_field!(model, A0)
    step!(model, warmup)

    ms = Float64[]
    Efin = 0.0; mfin = 0.0
    for rep in 1:reps
        apply_field!(model, A0)            # reinicia al campo teselado (estado inicial identico, fuera del cronometro)

        if save_photos && rep == 1
            mkpath(PHOTODIR)
            Representation._save_heatmap_photo(state_matrix(model), :viridis,
                "Cell_IA $(dims)x$(dims) — paso 0", joinpath(PHOTODIR, "cellia_$(dims)_step0.png"))
        end

        # ---------- REGION CRONOMETRADA: solo la simulacion ----------
        t0 = time_ns()
        step!(model, steps)
        dt_s = (time_ns() - t0) / 1e9
        # -------------------------------------------------------------

        push!(ms, 1000.0 * dt_s / steps)
        Efin = energy(model); mfin = maxstate(model)

        if save_photos && rep == 1
            Representation._save_heatmap_photo(state_matrix(model), :viridis,
                "Cell_IA $(dims)x$(dims) — paso $steps", joinpath(PHOTODIR, "cellia_$(dims)_final.png"))
        end
    end
    return ms, Efin, mfin
end

function main()
    quick = get(ENV, "BENCH_QUICK", "0") == "1"
    grids  = haskey(ENV, "BENCH_GRIDS") ? parse.(Int, split(ENV["BENCH_GRIDS"], ",")) : (quick ? [128] : [128, 256, 512])
    steps  = parse(Int, get(ENV, "BENCH_STEPS",  string(quick ? 50 : 1000)))
    warmup = parse(Int, get(ENV, "BENCH_WARMUP", string(quick ? 10 : 50)))
    reps   = parse(Int, get(ENV, "BENCH_REPS",   string(quick ? 2 : 10)))

    seed = read_seed(joinpath(@__DIR__, "orbium_seed.txt"))
    println("== Cell_IA (Agents.jl + FFTW) | hilos=$(FFTW.get_num_threads()) | steps=$steps reps=$reps ==")
    for dims in grids
        ms, Efin, mfin = run_grid(dims, seed, steps, warmup, reps)
        med = median(ms); sd = length(ms) > 1 ? std(ms) : 0.0
        total = med * steps / 1000
        @printf("  %4dx%-4d  %8.3f ms/step (+/-%.3f)  total=%.3f s  E=%.6f  max=%.6f\n",
                dims, dims, med, sd, total, Efin, mfin)
        open(RESULTS, "a") do io
            for (rep, v) in enumerate(ms)
                @printf(io, "cell_ia,Agents.jl+FFTW,CPU,%d,%d,%.6f,%.6f,%.9f,%.9f\n",
                        dims, rep, v, v * steps / 1000, Efin, mfin)
            end
        end
    end
end

main()
