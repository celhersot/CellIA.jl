module EvolutionRulesDefinition

using Agents

export gol_step_asyn!, gol_step_syn!, gol_model_step!, lenia_step!, predator_prey_step!, rps_step_syn!

function hola()
end
# --- GAME OF LIFE ---

# Asynchronous agents update for GoL
function gol_step_asyn!(agent, model)
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

# Synchronous agents update for GoL
function gol_step_syn!(agent, model)
    n_alive = count(n -> n.state == true, nearby_agents(agent, model))
    
    if agent.state == true
        if (n_alive >= model.min_to_live && n_alive <= model.max_to_live)
            agent.future_state = true
        else
            agent.future_state = false
        end
    else
        if n_alive == model.max_to_live
            agent.future_state = true
        else
            agent.future_state = false
        end
    end
end

# Auxiliar function to execute syn simulation of GoL
function gol_model_step!(model)
    for agent in allagents(model)
        agent.state = agent.future_state
    end
end

# --- PREDATOR - PREY ---

function predator_prey_step!(agent, model)
    # To implement.
end

# --- ROCK, PAPER, SCISSORS ---

function rps_step_syn!(agent, model)
    beaten_by = (agent.state % 3) + 1
    n_predators = count(n -> n.state == beaten_by, nearby_agents(agent, model))
    
    if n_predators >= model.threshold
        agent.future_state = beaten_by
    else
        agent.future_state = agent.state
    end
end

end