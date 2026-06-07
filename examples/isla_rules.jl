function isla_step!(agent, model)
    if agent.state == :tree
        n_water = count(nb -> nb.state == :water, nearby_agents(agent, model))
        if n_water > 0
            model.next_states[agent.id] = :water
        end
    elseif agent.state == :water
        n_tree = count(nb -> nb.state == :tree, nearby_agents(agent, model))
        if n_tree > 0
            model.next_states[agent.id] = :tree
        end
    else
        model.next_states[agent.id] = agent.state
    end
end
