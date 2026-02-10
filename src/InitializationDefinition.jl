module InitializationDefinition

export initialize_model

include("../src/AgentDefinition.jl")
include("../src/EvolutionRulesDefinition.jl")

using Agents
using Random
using .AgentDefinition
using .EvolutionRulesDefinition

function initialize_model(config::Dict)
    # Space definition (Discrete or Continuous)
    space_config = config["space"]
    space = nothing
    
    if space_config["type"] == "grid"
        dims = Tuple(space_config["dimensions"])
        space = GridSpaceSingle(dims; periodic = space_config["periodic"])
    elseif space_config["type"] == "continuous"
        space = ContinuousSpace(Tuple(space_config["dimensions"]))
    end

    # Global properties definition
    properties = Dict(Symbol(k) => v for (k, v) in config["properties"])

    # Getting the evolution rules
    rules_config = config["rules"]
    agent_step_fn = dummystep
    model_step_fn = dummystep

    if haskey(rules_config, "agent_step")
        fn_name = rules_config["agent_step"]
        agent_step_fn = getfield(EvolutionRulesDefinition, Symbol(fn_name))
    end
    if haskey(rules_config, "model_step")
        fn_name = rules_config["model_step"]
        model_step_fn = getfield(EvolutionRulesDefinition, Symbol(fn_name))
    end
    if haskey(rules_config, "evolution_rule")
        fn_name = rules_config["evolution_rule"]
        agent_step_fn = getfield(EvolutionRulesDefinition, Symbol(fn_name))
    end

    # Model definition
    rng = Xoshiro(config["simulation"]["seed"])
    
    model = StandardABM(
        UniversalAgent, 
        space;
        agent_step! = agent_step_fn,
        model_step! = model_step_fn,
        properties = properties,
        rng = rng,
        scheduler = Schedulers.Randomly()
    )

    # Populate the world
    populate_world!(model, config)

    return model
end

function populate_world!(model, config)
    for pos in positions(model)
        add_agent!(pos, model; state = false, future_state = false, group = 1)
    end

    rules_cfg = config["rules"]
    agent_cfg = config["agents"]
    init_strategy = get(rules_cfg, "initialization_rule", "random_placement")
    
    dims = size(abmspace(model))
    total_slots = prod(dims)
    
    total_agents_to_live = haskey(agent_cfg, "total_count") && agent_cfg["total_count"] > 0 ? 
                           agent_cfg["total_count"] : 
                           floor(Int, total_slots * agent_cfg["density"])

    if init_strategy == "random_placement"
        all_ids = collect(allids(model))
        shuffle!(abmrng(model), all_ids)
        for i in 1:total_agents_to_live
            target_id = all_ids[i]
            model[target_id].state = true
            model[target_id].future_state = true
        end

    elseif init_strategy == "sorted_placement_h"
        mid_y = div(dims[2], 2)
        center_x = div(dims[1], 2)
        half_len = div(total_agents_to_live, 2)
        
        for i in 1:total_agents_to_live
            x = clamp(center_x - half_len + i, 1, dims[1])
            id = id_in_position((x, mid_y), model)
            if id != 0
                model[id].state = true
                model[id].future_state = true
            end
        end

    elseif init_strategy == "sorted_placement_v"
        mid_x = div(dims[1], 2)
        center_y = div(dims[2], 2)
        half_len = div(total_agents_to_live, 2)

        for i in 1:total_agents_to_live
            y = clamp(center_y - half_len + i, 1, dims[2])
            id = id_in_position((mid_x, y), model)
            if id != 0
                model[id].state = true
                model[id].future_state = true
            end
        end
    end
end

end