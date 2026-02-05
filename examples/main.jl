import Pkg
Pkg.add("CairoMakie")
using Agents, Random, CairoMakie

# Cargar los módulos
include("../src/AgentDefinition.jl")
include("../src/EvolutionRules.jl")
include("../src/Initialization.jl")
include("../src/Representation.jl")

using .AgentDefinition
using .EvolutionRules
using .Initialization
using .Representation

# Configuración de parámetros
g_size = (50, 50)
p_live_agents = 0.30
nb_total_agents = trunc(Int, (g_size[1] * g_size[2]) * p_live_agents)

# Inicializar el modelo
game_of_life = initialize(
    total_agents = nb_total_agents, 
    gridsize = g_size, 
    min_to_live = 2, 
    max_to_live = 3, 
    seed = 125
)

println(typeof(game_of_life))

#                   model/fle_name/sim_title/agent_shape/agent_size/sim_framerate/sim_frames
video_simulation(game_of_life, "gol.mp4", "Game of Life", :rect, 13, 9, 50, :red, :blue)

