# =============================================================================
#  Fotos del ESTADO INICIAL de Cell_IA para cada tamaño de rejilla del benchmark
# =============================================================================
# Para cada rejilla (128, 256, 512) construye el modelo Cell_IA, aplica el mismo
# campo inicial que usa el benchmark (orbiums teselados a densidad constante) y
# guarda una foto del estado en el paso 0 en  output_fotos/grid<M>x<N>.png.
#
# No cronometra ni avanza la simulacion: solo el estado inicial. Reutiliza el
# motor y la inicializacion del framework (es lo que tiene Cell_IA en el paso 0).
#
# Uso:  julia bench/snapshot_initial.jl
#       BENCH_GRIDS=128,256,512 julia bench/snapshot_initial.jl
# =============================================================================
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "spaces", "HexagonalSpace.jl"))
include(joinpath(@__DIR__, "..", "src", "UniversalAgents.jl"))
include(joinpath(@__DIR__, "..", "src", "CustomEvolutionRules.jl"))
include(joinpath(@__DIR__, "..", "src", "SpaceDefinition.jl"))
include(joinpath(@__DIR__, "..", "src", "Initialization.jl"))
include(joinpath(@__DIR__, "..", "src", "Representation.jl"))

using .Initialization
using Agents
using CairoMakie

const PITCH = 64
const OUTDIR = joinpath(@__DIR__, "..", "output_fotos")

function make_config(dims::Int)
    Dict(
        "simulation" => Dict("model_name" => "snapshot_$(dims)", "seed" => 42),
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

apply_field!(model, A) = (for a in allagents(model); a.state = A[a.pos[1], a.pos[2]]; end; model)

function state_matrix(model)
    dims = size(abmspace(model))
    A = zeros(Float64, dims...)
    for a in allagents(model); A[a.pos[1], a.pos[2]] = Float64(a.state); end
    return A
end

# Heatmap limpio (sin titulo ni ejes), apto para figura de la memoria.
function save_clean_heatmap(A, path)
    fig = Figure(size = (700, 700), backgroundcolor = :black)
    ax  = Axis(fig[1, 1]; backgroundcolor = :black, aspect = DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, A; colormap = :viridis, colorrange = (0.0, 1.0))
    mkpath(dirname(path))
    save(path, fig)
    println("Guardada: $path")
end

function main()
    grids = haskey(ENV, "BENCH_GRIDS") ? parse.(Int, split(ENV["BENCH_GRIDS"], ",")) : [128, 256, 512]
    seed  = read_seed(joinpath(@__DIR__, "orbium_seed.txt"))
    for dims in grids
        model = initialize_model(make_config(dims))    # construye el mundo Cell_IA (paso 0)
        apply_field!(model, tiled_field(dims, seed))    # mismo campo inicial que el benchmark
        save_clean_heatmap(state_matrix(model),
                           joinpath(OUTDIR, "grid$(dims)x$(dims).png"))
    end
end

main()
