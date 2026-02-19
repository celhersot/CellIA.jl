using Agents
using Random

# Función paso de agente (ejemplo Forest Fire)
function forest_step!(agent, model)
    if agent.state == "tree"
        # Regla: Si hay un vecino "fire", me quemo
        for neighbor in nearby_agents(agent, model)
            if neighbor.state == "fire"
                agent.future_state = "fire"
                break
            end
        end
    elseif agent.state == "fire"
        # Regla: El fuego se consume y se convierte en ceniza
        agent.future_state = "ash"
    elseif agent.state == "ash"
        # Se queda como ceniza (o podrías hacer que vuelva a crecer)
        agent.future_state = "ash"
    end
end

# Función model_step (si fuera síncrona o necesitara update global)
function forest_model_step!(model)
    # Aplicar cambios de estado síncronos
    for agent in allagents(model)
        agent.state = agent.future_state
    end
end