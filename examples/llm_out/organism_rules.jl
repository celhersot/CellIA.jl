function organism_init!(model::Model)
    # Implement the initialization logic for the organism here
    # For example, set the initial state of all cells to a random value
    for cell in abmcells(model)
        cell.state = rand(model.rng)
    end
end
