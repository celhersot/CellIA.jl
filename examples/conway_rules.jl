function gol_step!(agent, model)
    n_live = count(nb -> nb.state == true, nearby_agents(agent, model))
    if agent.state == true
        model.next_states[agent.id] = n_live > 1 && n_live < 4 ? true : false
    else
        model.next_states[agent.id] = n_live == 3 ? true : false
    end
end
