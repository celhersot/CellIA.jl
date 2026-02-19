module CustomEvolutionRules
using Agents
using Random

# --- DEFAULT MODEL STEP ---

function default_model_step!(model)
    for agent in allagents(model)
        agent.state = agent.future_state
    end
end

# --- GAME OF LIFE ---

function gol_step!(agent, model)
    alives = 0
    for neighbor in nearby_agents(agent, model)
        if neighbor.state == "alive"
            alives += 1
        end
    end
    if agent.state == "alive"
        agent.future_state = (alives == model.min_to_live || alives == model.max_to_live) ? "alive" : "dead"
    else
        agent.future_state = (alives == model.max_to_live) ? "alive" : "dead"
    end
end

# --- ROCK, PAPER, SCISSORS ---

function rps_step!(agent, model)
    current_val = parse(Int, string(agent.state)) 
    beaten_by = (current_val % 3) + 1
    n_predators = count(n -> parse(Int, string(n.state)) == beaten_by, nearby_agents(agent, model))
    
    if n_predators >= model.threshold
        agent.future_state = beaten_by
    else
        agent.future_state = agent.state
    end
end

### --- SCHELLING'S SEGREGATION MODEL ---

function schelling_step!(agent, model)
    # 1. Convertimos a string para asegurar que "0" y 0 funcionen igual
    curr_state = string(agent.state)
    
    # Si soy una celda vacía, me aseguro de que mi futuro sea seguir vacío 
    # (a menos que alguien me sobrescriba)
    if curr_state == "0" || curr_state == "empty"
        # Solo actualizo si nadie me ha reclamado aún en este turno
        if string(agent.future_state) == curr_state
            agent.future_state = agent.state 
        end
        return
    end

    # 2. Contar vecinos
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

    # 3. Lógica de mudanza
    if count_neighbors > 0 && (count_identical / count_neighbors) < model.min_identical
        # Buscar celdas vacías QUE AÚN NO HAYAN SIDO RECLAMADAS en este step
        empty_cells = filter(
            a -> (string(a.state) == "0" || string(a.state) == "empty") && 
                 a.future_state == a.state, 
            collect(allagents(model))
        )
        
        if !isempty(empty_cells)
            target_cell = rand(abmrng(model), empty_cells)
            
            # Intercambio: Yo me vuelvo vacío, la celda vacía se vuelve yo
            agent.future_state = target_cell.state 
            target_cell.future_state = agent.state
        else
            # Si no hay sitio, me aguanto
            agent.future_state = agent.state
        end
    else
        # Estoy contento, me quedo igual
        agent.future_state = agent.state
    end
end

# --- LENIA ---
# --- Lenia (Continuous Cellular Automata) ---

function lenia_step!(agent, model)
    # parse(Float64, string(...)) es el "combo" más seguro en Julia
    current_val = parse(Float64, string(agent.state))
    
    neighbors = collect(nearby_agents(agent, model))
    
    if isempty(neighbors)
        avg = 0.0
    else
        # Sumamos convirtiendo cada vecino a Float64 con la misma técnica
        total = sum(parse(Float64, string(n.state)) for n in neighbors)
        avg = total / length(neighbors)
    end

    # Parámetros del modelo (estos sí deberían venir como números del TOML)
    μ, σ = model.lenia_mu, model.lenia_sigma
    
    # Función de crecimiento (campana de Gauss)
    # G(n) = 2 * exp( -((n-μ)² / 2σ²) ) - 1
    diff = avg - μ
    growth = 2.0 * exp(-( (diff^2) / (2.0 * σ^2) )) - 1.0

    # Actualización: nuevo = viejo + dt * crecimiento
    new_val = current_val + (model.dt * growth)
    agent.future_state = clamp(new_val, 0.0, 1.0)
end

# Ejemplo: Predator Prey
function predator_prey_step!(agent, model)
    # ...
end


end