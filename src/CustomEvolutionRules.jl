module CustomEvolutionRules
using Agents
using Random

# --- DEFAULT MODEL STEP ---
"""
function default_model_step!(model)
    for agent in allagents(model)
        agent.state = agent.future_state
    end
end
"""
function default_model_step!(model)
    for agent in allagents(model)
        if haskey(model.next_states, agent.id)
            agent.state = model.next_states[agent.id]
        end
    end
    empty!(model.next_states)
end

# --- GAME OF LIFE ---
"""
function gol_step!(agent, model)
    alives = 0
    for neighbor in nearby_agents(agent, model)
        if neighbor.state == true
            alives += 1
        end
    end
    if agent.state == true
        agent.future_state = (alives == model.min_to_live || alives == model.max_to_live) ? true : false
    else
        agent.future_state = (alives == model.max_to_live) ? true : false
    end
end
"""
function gol_step!(agent, model)
    alives = count(n -> n.state == true, nearby_agents(agent, model))
    
    if agent.state == true
        model.next_states[agent.id] = (alives == model.min_to_live || alives == model.max_to_live)
    else
        model.next_states[agent.id] = (alives == model.max_to_live)
    end
end

# --- ROCK, PAPER, SCISSORS ---

function rps_step!(agent, model)
    beats = Dict(:rock => :paper, :paper => :scissors, :scissors => :rock)
    predator_type = beats[agent.state]
    
    n_predators = count(n -> n.state == predator_type, nearby_agents(agent, model))
    
    if n_predators >= model.threshold
        model.next_states[agent.id] = predator_type
    else
        model.next_states[agent.id] = agent.state
    end
end

### --- SCHELLING'S SEGREGATION MODEL ---

function schelling_step!(agent, model)
    curr_state = string(agent.state)
    
    if curr_state == "0" || curr_state == "empty"
        if string(agent.future_state) == curr_state
            agent.future_state = agent.state 
        end
        return
    end

    count_neighbors = 0
    count_identical = 0
    
    for neighbor in nearby_agents(agent, model)
        n_state = string(neighbor.state)
        if n_state != "0" && n_state != "empty"
            count_neighbors += 1
            if n_state == curr_state
                count_identical += 1
            end
        end
    end

    if count_neighbors > 0 && (count_identical / count_neighbors) < model.min_identical
        empty_cells = filter(
            a -> (string(a.state) == "0" || string(a.state) == "empty") && 
                 a.future_state == a.state, 
            collect(allagents(model))
        )
        
        if !isempty(empty_cells)
            target_cell = rand(abmrng(model), empty_cells)
            
            agent.future_state = target_cell.state 
            target_cell.future_state = agent.state
        else
            agent.future_state = agent.state
        end
    else
        agent.future_state = agent.state
    end
end

# --- LENIA ---
# --- Lenia (Continuous Cellular Automata) ---

function lenia_step!(agent, model)
    current_val = parse(Float64, string(agent.state))
    
    neighbors = collect(nearby_agents(agent, model))
    
    if isempty(neighbors)
        avg = 0.0
    else
        total = sum(parse(Float64, string(n.state)) for n in neighbors)
        avg = total / length(neighbors)
    end

    μ, σ = model.lenia_mu, model.lenia_sigma
    
    diff = avg - μ
    growth = 2.0 * exp(-( (diff^2) / (2.0 * σ^2) )) - 1.0

    new_val = current_val + (model.dt * growth)
    agent.future_state = clamp(new_val, 0.0, 1.0)
end

# --- PREDATOR PREY --- IMPLEMENTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR
function predator_prey_step!(agent, model)
    # ...
end


end