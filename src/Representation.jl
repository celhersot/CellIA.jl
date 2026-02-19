module Representation
using CairoMakie
using Agents
using Random 
export video_simulation

function video_simulation(model, viz_config)
    color_scheme_in = get(viz_config, "color_scheme", nothing)
    var_name = get(viz_config, "variable_to_color", "state")
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    function resolve_color(agent)
        # 1. Obtener valor actual
        raw_val = hasfield(typeof(agent), Symbol(var_name)) ? getfield(agent, Symbol(var_name)) : agent.metadata[var_name]
        
        # --- CASO ESPECIAL: LENIA / CONTINUO (Float) ---
        # Si el valor es un número decimal (como en Lenia), usamos el gradiente
        val_float = tryparse(Float64, string(raw_val))
        
        if !isnothing(val_float) && (raw_val isa AbstractFloat || color_scheme_in == "viridis")
            # Gradiente de Azul (bajo) a Amarillo (alto)
            r = Int(floor(255 * val_float))
            g = Int(floor(255 * val_float))
            b = Int(floor(255 * (1.0 - val_float)))
            return "rgb($r, $g, $b)"
        end

        # --- CASO DISCRETO: (Estados como "1", "rock", :alive) ---
        if isa(color_scheme_in, Dict)
            return get(color_scheme_in, string(raw_val), "gray")
        
        elseif isa(color_scheme_in, Vector) && isa(raw_val, Int)
             idx = mod1(raw_val, length(color_scheme_in))
             return color_scheme_in[idx]
             
        else
            if isa(raw_val, Int)
                return fallback_colors[mod1(raw_val, length(fallback_colors))]
            else
                idx = mod1(hash(raw_val), length(fallback_colors))
                return fallback_colors[idx]
            end
        end
    end

    # --- Configuración de Formas ---
    shape_conf = get(viz_config, "agent_shape", "rect")
    function resolve_shape(agent)
        if isa(shape_conf, Dict)
             val = string(agent.state)
             return Symbol(get(shape_conf, val, "rect"))
        else
             return Symbol(shape_conf)
        end
    end

    # --- Generación ---
    abmvideo(
        viz_config["filename"], model;
        agent_color = resolve_color,
        agent_marker = resolve_shape,
        agent_size = get(viz_config, "agent_size", 12),
        framerate = get(viz_config, "framerate", 10),
        frames = get(viz_config, "frames", 50),
        title = get(viz_config, "title", "Simulation")
    )
end
end