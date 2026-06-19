function bee_step!(agent, model)
    neighbors = nearby_positions(agent.pos, model)
    if !isempty(neighbors)
        move_agent!(agent, rand(abmrng(model), neighbors), model)
    end
    props = abmspace(model).cell_properties[agent.pos]
    props[:honey] = get(props, :honey, 0.0) + 0.5
end
