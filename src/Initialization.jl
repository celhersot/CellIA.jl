module Initialization

using Agents
using Random
using ..UniversalAgents
using ..SpaceDefinition
using ..CustomEvolutionRules

export initialize_model

function initialize_model(config::Dict)
    # Creating the space with SpaceDefinition module
    space = create_space(config)

    # Global properties
    properties = Dict{Symbol, Any}()
    if haskey(config, "properties")
        for(k, v) in config["properties"]
            properties[Symbol(k)] = v
        end
    end

    # Loading rules from CustomEvolutionRules
    rules_conf = config["rules"]
    agent_step = dummystep
    model_step = dummystep

    if haskey(rules_conf, "agent_step")
        agent_step = getfield(CustomEvolutionRules, Symbol(rules_conf["agent_step"]))
    end
    if haskey(rules_conf, "model_step")
        model_step = getfield(CustomEvolutionRules, Symbol(rules_conf["model_step"]))
    end
    
    rng = Xoshiro(get(config["simulation"], "seed", 42))

    # Creating the model with UniversalAgent
    model = StandardABM(
        UniversalAgent,
        space;
        agent_step! = agent_step,
        model_step! = model_step,
        properties = properties,
        rng = rng,
        scheduler = Schedulers.Randomly()
    )

    # Populating the world
    populate_world!(model, config)

    return model
end

function populate_world!(model, config)
    agents_conf = config["agents"]
    pop_conf = agents_conf["population"]
    default_meta = get(agents_conf, "metadata", Dict{String, Any}())

    # Cell states
    dims = size(abmspace(model))
    total_slots = prod(dims)
    grid_states = fill(:empty, total_slots) # temporal

    init_rule = get(config["rules"], "initialization_rule", "random")

    states_to_spawn = []

    # Density or quantity?
    if haskey(pop_conf, "pop_density")
        for (state, density) in pop_conf["pop_density"]
            qty = floor(Int, total_slots * density)
            push!(states_to_spawn, (state, qty))
        end
    elseif haskey(pop_conf, "pop_quantity")
        for(state, qty) in pop_conf["pop_quantity"]
            push!(states_to_spawn, (state, qty))
        end
    end

    # Gererating positions
    available_indices = collect(1:total_slots)

    if init_rule == "random"
        shuffle!(abmrng(model), available_indices)

        current_idx = 1
        for (state_val, qty) in states_to_spawn
            for _ in 1:qty
                if current_idx <= length(available_indices)
                    flat_idx = available_indices[current_idx]
                    pos = flat_to_pos(flat_idx, dims)
                    add_agent!(pos, model; state = state_val, future_state = state_val, metadata = copy(default_meta))
                    current_idx += 1
                end
            end
        end
    elseif init_rule == "center_filled"
        # Fill from the center outwards
        center = dims .÷ 2
        sort!(available_indices, by = idx -> sum(abs.(flat_to_pos(idx, dims) .- center)))

        current_idx = 1
        for (state_val, qty) in states_to_spawn
            for _ in 1:qty
                if current_idx <= length(available_indices)
                    flat_idx = available_indices[current_idx]
                    pos = flat_to_pos(flat_idx, dims)
                    add_agent!(pos, model; state=state_val, future_state=state_val, metadata=copy(default_meta))
                    current_idx += 1
                end
            end
        end

        elseif init_rule == "horizontal_line"
            mid_y = dims[2] ÷ 2
            x_start = 1
            
            for (state_val, qty) in states_to_spawn
                for i in 0:(qty-1)
                    x = x_start + i
                    if x <= dims[1]
                        add_agent!((x, mid_y), model; state=state_val, future_state=state_val, metadata=copy(default_meta))
                    end
                end
                x_start += qty # Siguiente grupo tras el anterior
            end
        end
        # continuar
    end

function flat_to_pos(idx, dims)
    x = ((idx - 1) % dims[1]) + 1
    y = ((idx - 1) ÷ dims[1]) + 1
    return (x, y)
end
end
