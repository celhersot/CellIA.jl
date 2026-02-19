module UniversalAgents
using Agents
export UniversalAgent

@agent struct UniversalAgent(GridAgent{2})
    state::Union{Bool, Float64, Symbol, Int, String}
    future_state::Union{Bool, Float64, Symbol, Int, String} # For synchronous simulations
    #group::Int
    metadata::Dict{String, Any} # For define extra properties (group, energy, age...)
end

end