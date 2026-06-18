module Representation
using CairoMakie
using ColorTypes: Colorant
using Agents
using Random
using ..HexagonalSpace

export video_simulation, photo_simulation, run_simulation

const bird_polygon = Makie.Polygon(Point2f[(-1, -1), (2, 0), (-1, 1)])

function marker_shape(a)
    φ = atan(a.vel[2], a.vel[1])
    return rotate_polygon(bird_polygon, φ)
end

function get_agent_color(a, color_scheme)
    type_name = string(typeof(a).name.name)
    c = get(color_scheme, type_name, "gray")
    return Symbol(c)
end

const SPECIAL_FUNCTIONS = Dict(
    "marker" => marker_shape,
    "arrow"  => marker_shape,   # flecha orientada para agentes en espacio continuo
)

const _FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

function _build_agent_color_fn(viz_config)
    color_scheme = get(viz_config, "color_scheme", nothing)
    var_name     = get(viz_config, "variable_to_color", "state")

    # Si color_scheme es una paleta con nombre (p.ej. "viridis"), se precalcula una vez.
    cmap = nothing
    if isa(color_scheme, AbstractString) && !startswith(color_scheme, "#")
        try
            cmap = Makie.to_colormap(Symbol(color_scheme))
        catch
            cmap = nothing
        end
    end

    function resolve(agent)
        raw = agent
        try
            for part in split(string(var_name), ".")
                field = Symbol(part)
                raw = hasfield(typeof(raw), field) ? getfield(raw, field) : getproperty(raw, field)
            end
        catch
            raw = agent.state
        end

        # Estado que ya es un color: un Colorant, o cualquier struct con campos r/g/b.
        if raw isa Colorant
            return raw
        elseif hasproperty(raw, :r) && hasproperty(raw, :g) && hasproperty(raw, :b)
            return RGBf(clamp(Float64(getproperty(raw, :r)), 0.0, 1.0),
                        clamp(Float64(getproperty(raw, :g)), 0.0, 1.0),
                        clamp(Float64(getproperty(raw, :b)), 0.0, 1.0))
        end

        # Paleta con nombre + estado numerico: mapeo continuo a [0,1].
        if !isnothing(cmap) && isa(raw, AbstractFloat)
            t   = clamp(Float64(raw), 0.0, 1.0)
            idx = max(1, round(Int, t * (length(cmap) - 1) + 1))
            return cmap[idx]
        elseif isa(color_scheme, Dict)
            return get(color_scheme, string(raw), "gray")
        elseif isa(color_scheme, Vector) && isa(raw, Int)
            return color_scheme[mod1(raw, length(color_scheme))]
        else
            idx = isa(raw, Int) ? mod1(raw, length(_FALLBACK_COLORS)) : mod1(hash(raw), length(_FALLBACK_COLORS))
            return _FALLBACK_COLORS[idx]
        end
    end
    return resolve
end

function _build_agent_shape_fn(viz_config)
    shape_conf = get(viz_config, "agent_shape", "rect")
    function resolve(agent)
        isa(shape_conf, Dict) ? Symbol(get(shape_conf, string(agent.state), "rect")) : Symbol(shape_conf)
    end
    return resolve
end

# Color de celda hexagonal segun viz_config: cell_color_property (clave en cell_properties)
# y cell_color_max. Valor numerico = gradiente blanco-ambar; simbolico = color_scheme.
function _cell_color(cell_props::Dict{Symbol, Any}, viz_config)
    prop = get(viz_config, "cell_color_property", nothing)
    isnothing(prop) && return RGBAf(1.0, 1.0, 1.0, 1.0)

    val = get(cell_props, Symbol(prop), nothing)
    isnothing(val) && return RGBAf(1.0, 1.0, 1.0, 1.0)

    if isa(val, Number)
        max_val = Float64(get(viz_config, "cell_color_max", 1.0))
        t = clamp(Float64(val) / max_val, 0.0, 1.0)
        return RGBAf(1.0, 1.0 - 0.25*t, 1.0 - t, 1.0)
    else
        scheme = get(viz_config, "color_scheme", Dict())
        c = get(scheme, string(val), "white")
        return Makie.to_color(c)
    end
end

function _draw_hex_frame!(ax, model, viz_config, agent_color_fn)
    space = abmspace(model)
    dims  = space.dims

    for row in 1:dims[1], col in 1:dims[2]
        center  = hex_to_pixel(row, col)
        corners = Point2f.(hex_corners(center))
        poly!(ax, corners;
              color       = _cell_color(space.cell_properties[(row, col)], viz_config),
              strokecolor = RGBAf(0.15, 0.15, 0.15, 1.0),
              strokewidth = 1.0)
    end

    agent_size = Float64(get(viz_config, "agent_size", 15))
    gap = clamp(agent_size / 100.0, 0.3, 0.7)
    for agent in allagents(model)
        center  = hex_to_pixel(agent.pos[1], agent.pos[2])
        corners = Point2f.(hex_corners(center; gap=gap))
        poly!(ax, corners;
              color       = Makie.to_color(agent_color_fn(agent)),
              strokecolor = RGBAf(1.0, 1.0, 1.0, 0.6),
              strokewidth = 0.5)
    end
end

function record_hexagonal(model, viz_config)
    agent_color_fn = _build_agent_color_fn(viz_config)
    dims    = abmspace(model).dims
    s, m    = HEX_SIZE, HEX_SIZE * 1.5
    xmax    = s * sqrt(3) * (dims[2] - 0.5) + m
    ymax    = s * 1.5     * (dims[1] - 1)   + m
    frames  = get(viz_config, "frames", 50)
    fps     = get(viz_config, "framerate", 10)
    title   = get(viz_config, "title", "Simulation")
    output  = viz_config["filename"]

    fig = Figure(size=(820, 700), backgroundcolor=:white)
    ax  = Axis(fig[1, 1]; aspect=DataAspect(), backgroundcolor=:white)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -m, xmax)
    ylims!(ax, -m, ymax)

    println("Grabando $frames frames en $output...")
    record(fig, output, 1:frames; framerate=fps) do frame
        empty!(ax)
        ax.title = "$title - paso $frame"
        _draw_hex_frame!(ax, model, viz_config, agent_color_fn)
        step!(model, 1)
    end
    println("Guardado: $output")
end

function _save_heatmap_photo(A::Matrix{Float64}, cmap::Symbol, title::String, path::String)
    fig = Figure(size=(600, 600), backgroundcolor=:black)
    ax  = Axis(fig[1, 1]; backgroundcolor=:black, aspect=DataAspect(),
               title=title, titlecolor=:white)
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, A; colormap=cmap, colorrange=(0.0, 1.0))
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    save(path, fig)
    println("Foto guardada: $path")
end

# Heatmap para grids con estado float continuo y paleta con nombre (p.ej. Lenia).
# Claves opcionales en viz_config: photos (guardar foto inicial y final), photo_prefix.
function record_grid_heatmap(model, viz_config)
    cmap   = get(viz_config, "color_scheme", "viridis")
    frames = get(viz_config, "frames", 50)
    fps    = get(viz_config, "framerate", 10)
    title  = get(viz_config, "title", "Simulation")
    output = viz_config["filename"]
    dims   = size(abmspace(model))

    function _state_matrix()
        A = zeros(Float64, dims...)
        for agent in allagents(model)
            A[agent.pos[1], agent.pos[2]] = Float64(agent.state)
        end
        A
    end

    photos    = get(viz_config, "photos",       false)
    photo_pfx = get(viz_config, "photo_prefix", "output_photos/sim")
    cmap_sym  = Symbol(cmap)

    A_obs = Observable(_state_matrix())

    if photos
        _save_heatmap_photo(A_obs[], cmap_sym, title, "$(photo_pfx)_step0.png")
    end

    fig = Figure(size=(600, 600), backgroundcolor=:black)
    ax  = Axis(fig[1, 1]; backgroundcolor=:black, aspect=DataAspect(),
               title=title, titlecolor=:white)
    hidedecorations!(ax)
    hidespines!(ax)
    heatmap!(ax, A_obs; colormap=cmap_sym, colorrange=(0.0, 1.0))

    println("Grabando $frames frames en $output...")
    record(fig, output, 1:frames; framerate=fps) do _
        step!(model, 1)
        A_obs[] = _state_matrix()
    end
    println("Guardado: $output")

    if photos
        _save_heatmap_photo(A_obs[], cmap_sym, title, "$(photo_pfx)_final.png")
    end
end

# Render de grid para agentes que llevan color RGB propio (color_scheme = "rgb"): cada
# agente pinta su celda y las vacias muestran el fondo (color plano o background_image).
function _compose_rgb_canvas(model, base, colorfn, nx, ny)
    canvas = copy(base)
    @inbounds for a in allagents(model)
        x, y = a.pos
        if 1 <= x <= nx && 1 <= y <= ny
            canvas[x, y] = RGBf(colorfn(a))
        end
    end
    return canvas
end

function _rgb_background(model, viz_config, nx, ny)
    props = abmproperties(model)
    if get(viz_config, "show_target_background", false) && haskey(props, :background_image)
        return RGBf.(props[:background_image])
    end
    c = Makie.to_color(get(viz_config, "background_color", "black"))
    return fill(convert(RGBf, c), nx, ny)
end

function record_rgb_grid(model, viz_config)
    dims   = size(abmspace(model))
    nx, ny = dims
    frames = get(viz_config, "frames",    200)
    fps    = get(viz_config, "framerate",  20)
    title  = get(viz_config, "title", "Simulation")
    output = viz_config["filename"]

    colorfn = _build_agent_color_fn(viz_config)
    base    = _rgb_background(model, viz_config, nx, ny)
    canvas  = Observable(_compose_rgb_canvas(model, base, colorfn, nx, ny))

    fig = Figure(size = (640, 640), backgroundcolor = :black)
    ax  = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = :black,
               title = title, titlecolor = :white)
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, canvas; interpolate = false)

    dir = dirname(output)
    !isempty(dir) && mkpath(dir)
    println("Grabando $frames frames en $output...")
    record(fig, output, 1:frames; framerate = fps) do _
        step!(model, 1)
        canvas[] = _compose_rgb_canvas(model, base, colorfn, nx, ny)
    end
    println("Guardado: $output")
end

function video_simulation(model, viz_config, space_config)
    if space_config["type"] == "hexagonal"
        record_hexagonal(model, viz_config)
        return
    end

    color_scheme = get(viz_config, "color_scheme", nothing)

    # "rgb" se comprueba antes que el heatmap porque tambien es un string sin "#".
    if space_config["type"] == "grid" &&
            isa(color_scheme, AbstractString) && lowercase(color_scheme) == "rgb"
        record_rgb_grid(model, viz_config)
        return
    end

    # Grid + paleta con nombre: heatmap (p.ej. Lenia).
    if space_config["type"] == "grid" &&
            isa(color_scheme, AbstractString) && !startswith(color_scheme, "#")
        record_grid_heatmap(model, viz_config)
        return
    end

    if space_config["type"] == "continuous"
        shape_name = get(viz_config, "agent_shape", "rect")
        shape = haskey(SPECIAL_FUNCTIONS, shape_name) ? SPECIAL_FUNCTIONS[shape_name] : Symbol(shape_name)
        color = (a) -> get_agent_color(a, get(viz_config, "color_scheme", nothing))
    else
        color = _build_agent_color_fn(viz_config)
        shape = _build_agent_shape_fn(viz_config)
    end

    abmvideo(
        viz_config["filename"], model;
        agent_color  = color,
        agent_marker = shape,
        agent_size   = get(viz_config, "agent_size", 12),
        framerate    = get(viz_config, "framerate", 10),
        frames       = get(viz_config, "frames", 50),
        title        = get(viz_config, "title", "Simulation"),
    )
end

# Guarda una foto del estado actual. output_path: ruta donde guardar la figura.
function photo_simulation(model, viz_config, space_config,
                          output_path::Union{String, Nothing}=nothing)
    color_scheme = get(viz_config, "color_scheme", nothing)

    # Grid + color RGB por agente: compone un unico lienzo (igual que record_rgb_grid).
    if space_config["type"] == "grid" &&
            isa(color_scheme, AbstractString) && lowercase(color_scheme) == "rgb"
        nx, ny  = size(abmspace(model))
        colorfn = _build_agent_color_fn(viz_config)
        base    = _rgb_background(model, viz_config, nx, ny)
        canvas  = _compose_rgb_canvas(model, base, colorfn, nx, ny)
        fig = Figure(size = (640, 640), backgroundcolor = :black)
        ax  = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = :black,
                   title = get(viz_config, "title", "Simulation"), titlecolor = :white)
        hidedecorations!(ax); hidespines!(ax)
        image!(ax, canvas; interpolate = false)
        if !isnothing(output_path)
            dir = dirname(output_path); !isempty(dir) && mkpath(dir)
            save(output_path, fig)
            println("Foto guardada: $output_path")
        end
        return fig
    end

    # Heatmap para grids con estado float (paleta por defecto "viridis").
    first_ag = isempty(allagents(model)) ? nothing : first(allagents(model))
    state_is_float = !isnothing(first_ag) && isa(first_ag.state, AbstractFloat)
    use_heatmap = space_config["type"] == "grid" && state_is_float

    if use_heatmap
        cmap_str = (isa(color_scheme, AbstractString) && !startswith(color_scheme, "#")) ?
                   color_scheme : "viridis"
        dims = size(abmspace(model))
        A = zeros(Float64, dims...)
        for agent in allagents(model)
            A[agent.pos[1], agent.pos[2]] = Float64(agent.state)
        end
        title = get(viz_config, "title", "Simulation")
        if !isnothing(output_path)
            _save_heatmap_photo(A, Symbol(cmap_str), title, output_path)
            return nothing
        end
        fig = Figure(size=(600, 600), backgroundcolor=:black)
        ax  = Axis(fig[1, 1]; backgroundcolor=:black, aspect=DataAspect(),
                   title=title, titlecolor=:white)
        hidedecorations!(ax); hidespines!(ax)
        heatmap!(ax, A; colormap=Symbol(cmap_str), colorrange=(0.0, 1.0))
    else
        if space_config["type"] == "continuous"
            shape_name = get(viz_config, "agent_shape", "rect")
            shape = haskey(SPECIAL_FUNCTIONS, shape_name) ? SPECIAL_FUNCTIONS[shape_name] : Symbol(shape_name)
        else
            shape = _build_agent_shape_fn(viz_config)
        end
        fig, = abmplot(model; agent_marker=shape)
    end

    if !isnothing(output_path)
        dir = dirname(output_path)
        !isempty(dir) && mkpath(dir)
        save(output_path, fig)
        println("Foto guardada: $output_path")
    end
    return fig
end

# Ejecuta el modelo con run!, recoge datos por paso y escribe un CSV.
# Claves de [run]: steps, output (ruta CSV), adata, mdata, photos, photo_prefix.
function _write_csv(path::String, df)
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    open(path, "w") do f
        println(f, join(names(df), ","))
        for row in eachrow(df)
            println(f, join(values(row), ","))
        end
    end
end

# Resuelve una entrada de [run].adata / [run].mdata. Si el nombre coincide con una funcion
# de CustomEvolutionRules, devuelve la funcion (run! la muestrea cada paso como metrica
# calculada); si no, devuelve el Symbol (propiedad del modelo o campo del agente).
function _resolve_metric(key)
    sym = Symbol(key)
    if isdefined(Main.CustomEvolutionRules, sym)
        val = getfield(Main.CustomEvolutionRules, sym)
        val isa Function && return val
    end
    return sym
end

function run_simulation(model, run_config, viz_config, space_config)
    steps      = get(run_config, "steps",        100)
    output     = get(run_config, "output",        "output_data/results.csv")
    adata_keys = get(run_config, "adata",        ["state"])
    mdata_keys = get(run_config, "mdata",        String[])
    photos     = get(run_config, "photos",       false)
    photo_pfx  = get(run_config, "photo_prefix", "output_photos/sim")

    adata = isempty(adata_keys) ? nothing : [_resolve_metric(k) for k in adata_keys]
    mdata = isempty(mdata_keys) ? nothing : [_resolve_metric(k) for k in mdata_keys]

    if photos
        photo_simulation(model, viz_config, space_config, "$(photo_pfx)_step0.png")
    end

    println("Ejecutando $steps pasos...")
    adf, mdf = run!(model, steps; adata=adata, mdata=mdata)

    # Se prefieren los datos por agente; si no se pidieron, se guardan los del modelo.
    df_to_save = !isnothing(adata) ? adf : mdf
    if !isempty(df_to_save)
        _write_csv(output, df_to_save)
        println("Resultados guardados: $output")
    else
        println("No se recogieron datos (adata y mdata vacios).")
    end

    if photos
        photo_simulation(model, viz_config, space_config, "$(photo_pfx)_final.png")
    end
end

end
