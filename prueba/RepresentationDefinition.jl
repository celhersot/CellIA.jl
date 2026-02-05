module RepresentationDefinition
using CairoMakie, Agents
export video_simulation

function video_simulation(model, viz_config)
    # Extraer configuración
    color_map = viz_config["color_scheme"] # Dict("alive" => "green", "dead" => "yellow")
    var_name = viz_config["variable_to_color"] # "state"
    
    # Función de color dinámica
    function get_color(agent)
        # Obtenemos el valor del estado del agente (ej. true/false o 1/0)
        val = getfield(agent, Symbol(var_name)) 
        # Mapeamos ese valor al color definido en el TOML
        # Nota: requiere adaptar el TOML para que las claves coincidan con los valores del estado
        return val == true ? color_map["alive"] : color_map["dead"]
    end

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