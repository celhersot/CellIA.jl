function island_step!(agent, model)
    # Check if the cell is already white (island)
    if agent.state < 0.5
        # Check if all surrounding cells are white (no surrounding island)
        all_neighbours = collect(nearby_positions(agent.pos, model))
        if all(n -> n.state < 0.5, all_neighbours)
            # Spawn a new island at the cell
            model.next_states[agent.id] = 0.5
        end
    end
end