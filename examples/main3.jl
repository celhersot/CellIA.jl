import Pkg
Pkg.add("CairoMakie")
using Agents, Random, CairoMakie, TOML

# Cargar los módulos
include("../src/AgentDefinition.jl")
include("../src/EvolutionRules.jl")
include("../src/Initialization.jl")
include("../src/Representation.jl")

using .AgentDefinition
using .EvolutionRules
using .Initialization
using .Representation

# Función main que recibe archivo de config
function main(config_file::String="config.toml")
    config = TOML.parsefile(config_file)

    # Parámetros del modelo
    g_size = Tuple(config["model"]["g_size"])
    p_live_agents = config["model"]["p_live_agents"]
    nb_total_agents = trunc(Int, (g_size[1] * g_size[2]) * p_live_agents)
    min_to_live = config["model"]["min_to_live"]
    max_to_live = config["model"]["max_to_live"]
    seed = config["model"]["seed"]

    # Inicializar el modelo
    game_of_life = initialize(
        total_agents = nb_total_agents,
        gridsize = g_size,
        min_to_live = min_to_live,
        max_to_live = max_to_live,
        seed = seed
    )

    # Parámetros del video
    video_cfg = config["video"]
    video_simulation(
        game_of_life,
        video_cfg["filename"],
        video_cfg["title"],
        Symbol(video_cfg["agent_shape"]),
        video_cfg["agent_size"],
        video_cfg["sim_framerate"],
        video_cfg["sim_frames"],
        Symbol(video_cfg["color1"]),
        Symbol(video_cfg["color2"])
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
