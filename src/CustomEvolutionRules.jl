module CustomEvolutionRules
using Agents
using Random

# --- DEFAULT MODEL STEP ---

function default_model_step!(model)
    for agent in allagents(model)
        if haskey(model.next_states, agent.id)
            agent.state = model.next_states[agent.id]
        end
    end
    empty!(model.next_states)
end

# --- GAME OF LIFE ---

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
    neighbors = collect(nearby_agents(agent, model))
    
    # If alone -> happy
    if isempty(neighbors)
        return
    end

    same_group_count = count(n -> n.state == agent.state, neighbors)
    ratio = same_group_count / length(neighbors)
    
    # It moves
    if ratio < model.min_identical
        move_agent_single!(agent, model)
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