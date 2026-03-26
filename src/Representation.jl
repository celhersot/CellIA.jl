module Representation
using CairoMakie
using Agents
using Random
using .CustomSpaces
export video_simulation

const bird_polygon = Makie.Polygon(Point2f[(-1, -1), (2, 0), (-1, 1)])

function marker_shape(a)
    φ = atan(a.vel[2], a.vel[1]) #+ π/2 + π
    return rotate_polygon(bird_polygon, φ)
end

function get_agent_color(a, color_scheme)
    type_name = string(typeof(a).name.name)
    c = get(color_scheme, type_name, "gray")
    return Symbol(c)
end

const SPECIAL_FUNCTIONS = Dict(
    "marker" => marker_shape,
    # "drone_marker" => drone_marker_func, 
)

function video_simulation(model, viz_config, space_config)
    color_scheme_in = get(viz_config, "color_scheme", nothing)
    var_name = get(viz_config, "variable_to_color", "state")
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    if space_config["type"] == "continuous"
        shape_name = get(viz_config, "agent_shape", "rect")

        if haskey(SPECIAL_FUNCTIONS, shape_name)
            shape_name = SPECIAL_FUNCTIONS[shape_name]
        else
            shape_name = Symbol(shape_name)
        end

        shape = shape_name
        color = (a) -> get_agent_color(a, color_scheme_in)

    else
        
        function resolve_color(agent)
            # 1. Obtener valor actual
            raw_val = agent
            try
                for part in split(string(var_name), ".")
                    field = Symbol(part)
                    # Intentamos obtenerlo como campo (struct) o como propiedad (Agents.jl)
                    if hasfield(typeof(raw_val), field)
                        raw_val = getfield(raw_val, field)
                    else
                        raw_val = getproperty(raw_val, field)
                    end
                end
            catch e
                raw_val = agent.state
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

        color = resolve_color
        shape = resolve_shape
    end
        # --- Generación ---
    abmvideo(
        viz_config["filename"], model;
        agent_color = color,
        agent_marker = shape,
        agent_size = get(viz_config, "agent_size", 12),
        framerate = get(viz_config, "framerate", 10),
        frames = get(viz_config, "frames", 50),
        title = get(viz_config, "title", "Simulation")
    )
end

function Agents.space_axis_limits(model::ABM{<:HexagonalSpaceSingle})
    w, h = model.space.dims
end

function Agents.abmplot_pos(model::ABM{<:HexagonalSpaceSingle}, pos)
    q, r = pos
    x = sqrt(3) * (q + r/2.0)
    y = 1.5 * r
    return (Float32(x), Float32(y))
end
end