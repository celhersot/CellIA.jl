module RepresentationDefinition
using CairoMakie, Agents
export video_simulation

function video_simulation(model, viz_config)
    # Extract configuration
    color_map = viz_config["color_scheme"]
    var_name = viz_config["variable_to_color"]

    function get_color(agent)
        # Getting the state value of an agent
        val = getfield(agent, Symbol(var_name)) 
        # Mapping the color
        return val == true ? color_map["alive"] : color_map["dead"] # In the future, make it customizable.
    end

    # Create the simulation and export the video
    abmvideo(
        viz_config["filename"], model;
        agent_color = get_color,
        agent_marker = Symbol(viz_config["agent_shape"]),
        agent_size = viz_config["agent_size"],
        framerate = viz_config["framerate"],
        frames = viz_config["frames"],
        title = viz_config["title"]
    )
end
end