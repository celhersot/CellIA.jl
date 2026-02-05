module EvolutionRulesDefinition

using Agents
export gol_step!, lenia_step!, predator_prey_step!

# --- JUEGO DE LA VIDA ---
function gol_step!(agent, model)
    # Nota: ahora accedemos a propiedades genéricas del modelo
    n_alive = count(n -> n.state == true, nearby_agents(agent, model))
    
    if agent.state == true
        if !(n_alive >= model.min_to_live && n_alive <= model.max_to_live)
            agent.state = false
        end
    else
        if n_alive == model.max_to_live
            agent.state = true
        end
    end
end

# --- OTRO EJEMPLO (Simplificado) ---
function predator_prey_step!(agent, model)
    # Lógica de depredador presa...
end

end