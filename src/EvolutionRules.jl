module EvolutionRules

export gol_step!

using Agents

function gol_step!(agent, model)
    minlive = model.min_to_live
    maxlive = model.max_to_live
    count_neighbors_alive = 0

    # Contar vecinos vivos
    for neigh in nearby_agents(agent, model)
        count_neighbors_alive += neigh.status
    end

    # Reglas de Game of Life
    if agent.status == true
        if !(count_neighbors_alive >= minlive && count_neighbors_alive <= maxlive)
            agent.status = false
        end
    else
        if count_neighbors_alive == maxlive
            agent.status = true
        end
    end
end

end