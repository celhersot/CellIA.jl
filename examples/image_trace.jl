# Reglas de la simulacion "image trace": agentes de color que reproducen una imagen.
# Cada agente nace con un color RGB aleatorio, busca en su radio la celda cuyo color
# objetivo mas se parece al suyo y se mueve hacia ella. Con chameleon=true, al asentarse
# copia el color exacto de la imagen. Actualizacion sincrona, 1 agente por celda.

using FileIO
using ColorTypes

mutable struct Painter
    r::Float64
    g::Float64
    b::Float64
    found::Bool
    stuck::Int      # pasos seguidos sin poder avanzar
end

@inline _cdist(r1, g1, b1, r2, g2, b2) = (r1 - r2)^2 + (g1 - g2)^2 + (b1 - b2)^2

# Carga la imagen y la reescala al tamano de la rejilla (promediado por bloques),
# devolviendo los tres canales R,G,B en [0,1]. Sin imagen valida usa un patron de demo.
function _load_or_make_target(path::String, nx::Int, ny::Int)
    img = nothing
    if !isempty(path) && isfile(path)
        try
            img = FileIO.load(path)
        catch e
            @warn "No se pudo cargar la imagen '$path' ($e). Uso un patron sintetico."
            img = nothing
        end
    elseif !isempty(path)
        @warn "No existe la imagen '$path'. Uso un patron sintetico."
    end

    tr = Array{Float64}(undef, nx, ny)
    tg = Array{Float64}(undef, nx, ny)
    tb = Array{Float64}(undef, nx, ny)

    if img === nothing
        _fill_demo_target!(tr, tg, tb, nx, ny)
        return tr, tg, tb
    end

    H, W = size(img)
    @inbounds for x in 1:nx
        c0 = floor(Int, (x - 1) / nx * W) + 1
        c1 = max(c0, floor(Int, x / nx * W))
        for y in 1:ny
            # y crece hacia arriba; la fila 1 de la imagen es la de arriba (volteo vertical)
            r0 = floor(Int, (ny - y) / ny * H) + 1
            r1 = max(r0, floor(Int, (ny - y + 1) / ny * H))
            sr = 0.0; sg = 0.0; sb = 0.0; cnt = 0
            for r in r0:min(r1, H), c in c0:min(c1, W)
                px = img[r, c]
                sr += Float64(red(px)); sg += Float64(green(px)); sb += Float64(blue(px))
                cnt += 1
            end
            if cnt == 0
                px = img[min(r0, H), min(c0, W)]
                sr = Float64(red(px)); sg = Float64(green(px)); sb = Float64(blue(px)); cnt = 1
            end
            tr[x, y] = sr / cnt; tg[x, y] = sg / cnt; tb[x, y] = sb / cnt
        end
    end
    return tr, tg, tb
end

# Patron de demo cuando no hay imagen: cielo en degradado, sol y suelo.
function _fill_demo_target!(tr, tg, tb, nx, ny)
    cx, cy = nx * 0.5, ny * 0.58
    rad    = min(nx, ny) * 0.20
    @inbounds for x in 1:nx, y in 1:ny
        t = (y - 1) / max(1, ny - 1)
        r = 0.25 + 0.35 * t
        g = 0.45 + 0.35 * t
        b = 0.75 + 0.20 * t
        if y < ny * 0.25
            r, g, b = 0.20, 0.55, 0.22
        end
        if (x - cx)^2 + (y - cy)^2 <= rad^2
            r, g, b = 1.0, 0.72, 0.12
        end
        tr[x, y] = r; tg[x, y] = g; tb[x, y] = b
    end
    return nothing
end

# post_init: prepara la imagen objetivo, el fondo opcional y siembra los agentes.
function trace_init!(model)
    props  = abmproperties(model)
    dims   = size(abmspace(model))
    nx, ny = dims
    rng    = abmrng(model)

    path = string(get(props, :image_path, ""))
    tr, tg, tb = _load_or_make_target(path, nx, ny)
    props[:_trace_state] = (tr = tr, tg = tg, tb = tb, nx = nx, ny = ny)

    bg = Matrix{RGB{Float32}}(undef, nx, ny)
    @inbounds for x in 1:nx, y in 1:ny
        bg[x, y] = RGB{Float32}(tr[x, y], tg[x, y], tb[x, y])
    end
    props[:background_image] = bg

    density = Float64(get(props, :density, 0.7))
    n = clamp(round(Int, density * nx * ny), 1, nx * ny)
    coords = vec([(x, y) for x in 1:nx, y in 1:ny])
    shuffle!(rng, coords)
    @inbounds for k in 1:n
        pos = coords[k]
        add_agent!(pos, model; state = Painter(rand(rng), rand(rng), rand(rng), false, 0))
    end
    return nothing
end

function trace_model_step!(model)
    props    = abmproperties(model)
    st       = props[:_trace_state]
    R        = Int(get(props, :search_radius, 8))
    cham     = Bool(get(props, :chameleon, true))
    rate     = Float64(get(props, :chameleon_rate, 0.15))
    tol2     = Float64(get(props, :arrival_tol, 0.12))^2
    patience = Int(get(props, :patience, 3))
    _trace_run_step!(model, st, R, cham, rate, tol2, patience)
end

function _trace_run_step!(model, st, R::Int, cham::Bool, rate::Float64,
                          tol2::Float64, patience::Int)
    tr, tg, tb = st.tr, st.tg, st.tb
    nx, ny     = st.nx, st.ny

    agents = collect(allagents(model))
    n      = length(agents)
    n == 0 && return nothing

    # 1. Cada agente elige destino desde el estado actual (todos deciden a la vez):
    #    la mejor celda de color en su radio. Si ya esta en ella, se asienta.
    desired    = Vector{Tuple{Int,Int}}(undef, n)
    willsettle = Vector{Bool}(undef, n)
    @inbounds for i in 1:n
        a = agents[i]; s = a.state
        if s.found
            desired[i] = a.pos; willsettle[i] = true
            continue
        end
        px, py = a.pos
        bestx, besty = px, py
        bestd = _cdist(s.r, s.g, s.b, tr[px, py], tg[px, py], tb[px, py])
        x0 = max(1, px - R); x1 = min(nx, px + R)
        y0 = max(1, py - R); y1 = min(ny, py + R)
        for cx in x0:x1, cy in y0:y1
            d = _cdist(s.r, s.g, s.b, tr[cx, cy], tg[cx, cy], tb[cx, cy])
            if d < bestd
                bestd = d; bestx = cx; besty = cy
            end
        end
        if (bestx, besty) == (px, py) || bestd <= tol2
            desired[i] = (px, py); willsettle[i] = true
        else
            desired[i] = (px + sign(bestx - px), py + sign(besty - py))
            willsettle[i] = false
        end
    end

    # 2. Movimiento por pasadas: un agente entra a su destino solo si sigue vacio
    #    (asi nunca coinciden dos en la misma celda). Las pasadas dejan fluir cadenas
    #    de agentes; el orden aleatorio reparte los conflictos y es reproducible.
    rng   = abmrng(model)
    order = randperm(rng, n)
    moved = falses(n)
    maxpasses = 6
    pass = 0
    progress = true
    while progress && pass < maxpasses
        progress = false; pass += 1
        @inbounds for idx in order
            (moved[idx] || willsettle[idx]) && continue
            a = agents[idx]; d = desired[idx]
            d == a.pos && continue
            if isempty(d, model)
                move_agent!(a, d, model)
                moved[idx] = true
                progress = true
            end
        end
    end

    # 3. Asentamiento (por paciencia si lleva varios pasos bloqueado) y, si toca,
    #    transformacion camaleonica hacia el color objetivo de su celda.
    @inbounds for i in 1:n
        a = agents[i]; s = a.state
        if !s.found
            if willsettle[i]
                s.found = true
            elseif moved[i]
                s.stuck = 0
            else
                s.stuck += 1
                s.stuck >= patience && (s.found = true)
            end
        end
        if cham && s.found
            px, py = a.pos
            s.r += rate * (tr[px, py] - s.r)
            s.g += rate * (tg[px, py] - s.g)
            s.b += rate * (tb[px, py] - s.b)
        end
    end
    return nothing
end

# Metricas para [run]: error medio de color y fraccion de agentes asentados.
function mean_color_error(model)
    st = abmproperties(model)[:_trace_state]
    s = 0.0; n = 0
    for a in allagents(model)
        px, py = a.pos
        s += sqrt(_cdist(a.state.r, a.state.g, a.state.b,
                         st.tr[px, py], st.tg[px, py], st.tb[px, py]))
        n += 1
    end
    return n == 0 ? 0.0 : s / n
end

function found_fraction(model)
    n = nagents(model)
    return n == 0 ? 0.0 : count(a -> a.state.found, allagents(model)) / n
end
