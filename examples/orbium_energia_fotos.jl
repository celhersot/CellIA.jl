# Genera el video y tres fotos (inicial, mitad y final) del orbium en dos fases: sin
# perturbacion y con ruido estocastico, leyendo lenia_perturbation.toml/.jl.
#   julia examples/orbium_energia_fotos.jl
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
using CairoMakie

# Inyecta las reglas del experimento (orbium + lenia_perturbation_init!) en CustomEvolutionRules,
# igual que hace examples/main.jl con los archivos de reglas de usuario.
Base.include(Main.CustomEvolutionRules, abspath(joinpath(@__DIR__, "lenia_perturbation.jl")))

const FOTODIR  = joinpath(@__DIR__, "..", "output_fotos")
const VIDEODIR = joinpath(@__DIR__, "..", "output_videos")

function state_matrix(model)
    dims = size(abmspace(model))
    A = zeros(Float64, dims...)
    for a in allagents(model); A[a.pos[1], a.pos[2]] = Float64(a.state); end
    return A
end
energy(model) = sum(a.state for a in allagents(model))

function save_photo(A, title, path)
    fig = Figure(size = (640, 690), backgroundcolor = :black)
    ax  = Axis(fig[1, 1]; backgroundcolor = :black, aspect = DataAspect(),
               title = title, titlecolor = :white, titlesize = 22)
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, A; colormap = :viridis, colorrange = (0.0, 1.0))
    mkpath(dirname(path)); save(path, fig)
    println("Foto: $path")
end

function run_phase(base_config, key, label, sigma_noise)
    config = deepcopy(base_config)
    config["properties"]["sigma_noise"] = sigma_noise
    model = initialize_model(config)

    viz     = config["visualization"]
    nframes = Int(get(viz, "frames", 400))
    fps     = Int(get(viz, "framerate", 30))
    mid     = nframes ÷ 2

    # --- estado inicial (paso 0) ---
    A0 = state_matrix(model)
    save_photo(A0, "$label\npaso 0   E=$(round(energy(model); digits=1))",
               joinpath(FOTODIR, "orbium_energia_$(key)_inicial.png"))

    # --- video + capturas en mitad y final ---
    A_obs = Observable(A0)
    title_obs = Observable("$label   paso 0")
    fig = Figure(size = (640, 690), backgroundcolor = :black)
    ax  = Axis(fig[1, 1]; backgroundcolor = :black, aspect = DataAspect(),
               title = title_obs, titlecolor = :white, titlesize = 22)
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, A_obs; colormap = :viridis, colorrange = (0.0, 1.0))

    mkpath(VIDEODIR)
    vpath = joinpath(VIDEODIR, "orbium_energia_$(key).mp4")
    record(fig, vpath, 1:nframes; framerate = fps) do frame
        step!(model, 1)
        A = state_matrix(model)
        A_obs[] = A
        E = energy(model)
        title_obs[] = "$label   paso $frame   E=$(round(E; digits=1))"
        if frame == mid
            save_photo(A, "$label\npaso $mid   E=$(round(E; digits=1))",
                       joinpath(FOTODIR, "orbium_energia_$(key)_medio.png"))
        elseif frame == nframes
            save_photo(A, "$label\npaso $nframes   E=$(round(E; digits=1))",
                       joinpath(FOTODIR, "orbium_energia_$(key)_final.png"))
        end
    end
    println("Video: $vpath\n")
end

function main()
    base = TOML.parsefile(joinpath(@__DIR__, "lenia_perturbation.toml"))
    sigma = Float64(get(base["properties"], "sigma_noise", 0.04))

    println("Fase normal (sin perturbacion)")
    run_phase(base, "normal", "Normal (sin perturbacion)", 0.0)

    println("Fase con perturbacion (sigma_noise=$sigma)")
    run_phase(base, "perturbacion", "Con perturbacion (sigma=$sigma)", sigma)
end

main()
