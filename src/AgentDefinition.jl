module AgentDefinition

using Agents

export UniversalAgent

@agent struct UniversalAgent(GridAgent{2})
    state::Union{Bool, Float64, Symbol, Int}
    future_state::Union{Bool, Float64, Symbol, Int} # For synchronous simulations
    group::Int
end

end