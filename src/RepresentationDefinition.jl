module RepresentationDefinition
using CairoMakie, Agents
export video_simulation

function video_simulation(model, viz_config)
    color_map = viz_config["color_scheme"]
    var_name = viz_config["variable_to_color"]

    function get_color(agent)
        val = getfield(agent, Symbol(var_name)) 
        if val isa Bool
            key = val ? "alive" : "dead"
        else
            key = string(val)
        end

        return get(color_map, key, "gray")
    end

    abmvideo(
        viz_config["filename"], model;
        agent_color = get_color,
        agent_marker = Symbol(get(viz_config, "agent_shape", "rect")),
        agent_size = get(viz_config, "agent_size", 10),
        framerate = get(viz_config, "framerate", 10),
        frames = get(viz_config, "frames", 100),
        title = get(viz_config, "title", "ABM Simulation")
    )
end
end