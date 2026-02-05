module Initialization

export initialize

include("../src/EvolutionRules.jl")
include("../src/AgentDefinition.jl")

using .EvolutionRules
using .AgentDefinition
using Agents
using Random: Xoshiro

function initialize(; total_agents, gridsize, min_to_live, max_to_live, seed)
    space = GridSpaceSingle(gridsize; periodic = true)
    properties = Dict(:min_to_live => min_to_live, :max_to_live => max_to_live)
    rng = Xoshiro(seed)
    
    model = StandardABM(
        GoLAgent, space;
        agent_step! = gol_step!, 
        properties, 
        rng,
        container = Vector, 
        scheduler = Schedulers.Randomly()
    )

    for n in 1:total_agents
        add_agent_single!(model; status = true)
    end
    return model
end

end