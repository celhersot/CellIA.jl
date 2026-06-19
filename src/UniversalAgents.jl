module UniversalAgents
using Agents
export UniversalAgent

@agent struct UniversalAgent{T}(GridAgent{2})
    state::T
end

end