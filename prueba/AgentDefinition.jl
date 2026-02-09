module AgentDefinition
using Agents
export UniversalAgent

# Un agente que sirve para casi todo (Grid y Continuo)
@agent struct UniversalAgent(GridAgent{2})
    # status podría ser un Bool (GoL), un Float (Lenia) o un Symbol (Piedra papel tijeras)
    state::Union{Bool, Float64, Symbol, Int}
    future_state::Union{Bool, Float64, Symbol, Int} # For synchronous simulations
    group::Int # Para diferenciar tipos (ej. lobo vs oveja)
end

# Si necesitas Continuous, tendrías que definir otro o usar una macro que genere el struct
# Pero para empezar, usa GridAgent{2} como base.
end