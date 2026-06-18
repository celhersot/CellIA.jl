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
    "marker" => marker_shape,   # legacy alias
    "arrow"  => marker_shape,   # oriented arrow for continuous-space agents (reusable)
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

const _FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

function _build_agent_color_fn(viz_config)
    color_scheme = get(viz_config, "color_scheme", nothing)
    var_name     = get(viz_config, "variable_to_color", "state")

    # Precompute colormap once if color_scheme is a named palette string (e.g. "viridis").
    cmap = nothing
    if isa(color_scheme, AbstractString) && !startswith(color_scheme, "#")
        try
            cmap = Makie.to_colormap(Symbol(color_scheme))
        catch
            cmap = nothing   # fall through to Dict/fallback logic
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

        # State that *is* a color: a Colorant, or any struct exposing r/g/b fields.
        # Generic — lets any model carry a per-agent RGB and render it verbatim
        # (e.g. the image-tracing painters). Placed before the colormap/Dict logic;
        # primitive states (Int/Float/Symbol/String/Bool) have no r/g/b properties.
        if raw isa Colorant
            return raw
        elseif hasproperty(raw, :r) && hasproperty(raw, :g) && hasproperty(raw, :b)
            return RGBf(clamp(Float64(getproperty(raw, :r)), 0.0, 1.0),
                        clamp(Float64(getproperty(raw, :g)), 0.0, 1.0),
                        clamp(Float64(getproperty(raw, :b)), 0.0, 1.0))
        end

        # Named colormap + numeric state → continuous mapping into [0,1]
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

# ── Hexagonal rendering ────────────────────────────────────────────────────────
# Cell color is driven by viz_config keys:
#   cell_color_property  – key inside cell_properties to read (e.g. "honey")
#   cell_color_max       – numeric maximum for gradient normalisation (default 1.0)
# When the value is numeric → white-to-amber gradient.
# When symbolic/string    → looked up in color_scheme.

function _cell_color(cell_props::Dict{Symbol, Any}, viz_config)
    prop = get(viz_config, "cell_color_property", nothing)
    isnothing(prop) && return RGBAf(1.0, 1.0, 1.0, 1.0)

    val = get(cell_props, Symbol(prop), nothing)
    isnothing(val) && return RGBAf(1.0, 1.0, 1.0, 1.0)

    if isa(val, Number)
        max_val = Float64(get(viz_config, "cell_color_max", 1.0))
        t = clamp(Float64(val) / max_val, 0.0, 1.0)
        return RGBAf(1.0, 1.0 - 0.25*t, 1.0 - t, 1.0)   # white → amber
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

    println("Recording $frames frames → $output ...")
    record(fig, output, 1:frames; framerate=fps) do frame
        empty!(ax)
        ax.title = "$title — step $frame"
        _draw_hex_frame!(ax, model, viz_config, agent_color_fn)
        step!(model, 1)
    end
    println("Saved: $output")
end

# ── _save_heatmap_photo ────────────────────────────────────────────────────────

function _save_heatmap_photo(A::Matrix{Float64}, cmap::Symbol, title::String, path::String)
    fig = Figure(size=(600, 600), backgroundcolor=:black)
    ax  = Axis(fig[1, 1]; backgroundcolor=:black, aspect=DataAspect(),
               title=title, titlecolor=:white)
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, A; colormap=cmap, colorrange=(0.0, 1.0))
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    save(path, fig)
    println("Saved photo: $path")
end

# ── record_grid_heatmap ────────────────────────────────────────────────────────
# Used for grid simulations with a continuous float state and a named colormap
# (e.g. Lenia). Renders a full-coverage heatmap with black background.
# Optional keys in viz_config:
#   photos       – bool, save step-0 and final-frame photos (default false)
#   photo_prefix – path prefix for photo files (default "output_photos/sim")

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

    println("Recording $frames frames → $output ...")
    record(fig, output, 1:frames; framerate=fps) do _
        step!(model, 1)
        A_obs[] = _state_matrix()
    end
    println("Saved: $output")

    if photos
        _save_heatmap_photo(A_obs[], cmap_sym, title, "$(photo_pfx)_final.png")
    end
end

# ── record_rgb_grid ──────────────────────────────────────────────────────────
# Grid renderer for agents that carry a per-agent RGB color (color_scheme = "rgb").
# Each agent paints its own cell; empty cells show the background. The background is
# either a flat color (background_color, default black) or, when show_target_background
# is true and the model exposes a :background_image (Matrix of Colorant), that image.
# This is how the image-tracing model is visualized, but it is generic: any grid model
# whose agent state encodes a color renders here.

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
    println("Recording $frames frames → $output ...")
    record(fig, output, 1:frames; framerate = fps) do _
        step!(model, 1)
        canvas[] = _compose_rgb_canvas(model, base, colorfn, nx, ny)
    end
    println("Saved: $output")
end

# ── video_simulation ───────────────────────────────────────────────────────────

function video_simulation(model, viz_config, space_config)
    if space_config["type"] == "hexagonal"
        record_hexagonal(model, viz_config)
        return
    end

    color_scheme = get(viz_config, "color_scheme", nothing)

    # Grid + per-agent RGB color (each agent paints its own cell). Checked before the
    # heatmap branch because "rgb" is also a plain (non-"#") string.
    if space_config["type"] == "grid" &&
            isa(color_scheme, AbstractString) && lowercase(color_scheme) == "rgb"
        record_rgb_grid(model, viz_config)
        return
    end

    # Grid + named colormap → heatmap rendering (e.g. Lenia)
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

# ── photo_simulation ───────────────────────────────────────────────────────────
# output_path: if provided, saves the figure to that file.

function photo_simulation(model, viz_config, space_config,
                          output_path::Union{String, Nothing}=nothing)
    color_scheme = get(viz_config, "color_scheme", nothing)

    # Grid + per-agent RGB color → single composed canvas (mirrors record_rgb_grid).
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
            println("Saved photo: $output_path")
        end
        return fig
    end

    # Use heatmap for grid spaces whose state is a float, regardless of whether
    # color_scheme is set (default to "viridis" when not specified).
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
        println("Saved photo: $output_path")
    end
    return fig
end

# ── run_simulation ─────────────────────────────────────────────────────────────
# Drives the model with run!, collects per-step data, writes a CSV.
# Optionally saves a heatmap/plot at step 0 and at the final step.
#
# Relevant TOML keys in [run]:
#   steps        – number of steps to run (default 100)
#   output       – CSV output path (default "output_data/results.csv")
#   adata        – list of agent property names to collect (default ["state"])
#   mdata        – list of model property names to collect (default [])
#   photos       – bool, save photos at step 0 and final step (default false)
#   photo_prefix – path prefix for photo files (default "output_photos/sim")

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

# Resolve one [run].adata / [run].mdata entry.
# If the name matches a FUNCTION defined in CustomEvolutionRules, return the function
# object — Agents.jl run! then samples f(model) (mdata) or f(agent) (adata) every step,
# which is how a *computed* metric (total energy, entropy, active-cell count, …) is
# collected. Otherwise return a Symbol, i.e. a plain model property / agent field name
# (the previous behaviour). Generic: any model can declare a metric by naming a function
# in its rules file — nothing here is specific to Lenia or to "energy".
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

    println("Running $steps steps...")
    adf, mdf = run!(model, steps; adata=adata, mdata=mdata)

    # Prefer agent data; fall back to model data if no adata was requested.
    df_to_save = !isnothing(adata) ? adf : mdf
    if !isempty(df_to_save)
        _write_csv(output, df_to_save)
        println("Saved results: $output")
    else
        println("No data collected (both adata and mdata are empty).")
    end

    if photos
        photo_simulation(model, viz_config, space_config, "$(photo_pfx)_final.png")
    end
end

end
