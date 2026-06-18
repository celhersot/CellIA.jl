# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRACE — agentes de color que "calcan" una imagen.
#
# Idea: el usuario aporta una imagen. La rejilla es un lienzo de MENOR resolución
# que la imagen (para que la simulación no pese). Se siembran agentes con colores
# ALEATORIOS. Cada agente busca, dentro de su radio, la celda cuyo color objetivo
# (la imagen reescalada) más se parece al suyo y se desplaza UNA celda hacia ella
# (píxel a píxel). Cuando ya no puede mejorar (está en su mejor sitio) se da por
# "asentado".
#
#   • Simulación 1 (camaleón, chameleon = true):  el agente asentado va copiando el
#     color EXACTO de la imagen en su celda → el lienzo reproduce la imagen.
#   • Simulación 2 (pointillista, chameleon = false): el agente conserva su color
#     aleatorio y solo busca su sitio; el fondo muestra la imagen a copiar.
#
# Actualización SÍNCRONA y una sola ocupación por celda: todos deciden su movimiento
# a partir del MISMO estado; si varios quieren la misma celda, entra uno solo, y nunca
# a una celda ocupada (la rejilla GridSpaceSingle ya garantiza 1 agente/celda).
# ─────────────────────────────────────────────────────────────────────────────

using FileIO
using ColorTypes

# Estado del agente: su color RGB actual (∈[0,1]), si ya encontró su sitio y cuántos
# pasos lleva sin poder avanzar (para asentarse tras agotar la paciencia).
mutable struct Painter
    r::Float64
    g::Float64
    b::Float64
    found::Bool
    stuck::Int
end

# Distancia de color al cuadrado (evita la raíz en el bucle caliente).
@inline _cdist(r1, g1, b1, r2, g2, b2) = (r1 - r2)^2 + (g1 - g2)^2 + (b1 - b2)^2

# ── Carga / reescalado de la imagen objetivo ────────────────────────────────────
# Devuelve tres matrices nx×ny (canales R, G, B en [0,1]) ya reescaladas al tamaño
# de la rejilla mediante promediado por bloques (filtro de caja). Orientación pensada
# para image!: target[x,y] con y creciente hacia arriba ⇒ la imagen se ve derecha.
# Si la ruta está vacía o no se puede cargar, sintetiza un patrón de demostración.
function _load_or_make_target(path::String, nx::Int, ny::Int)
    img = nothing
    if !isempty(path) && isfile(path)
        try
            img = FileIO.load(path)
        catch e
            @warn "No se pudo cargar la imagen '$path' ($e). Uso un patrón sintético."
            img = nothing
        end
    elseif !isempty(path)
        @warn "No existe la imagen '$path'. Uso un patrón sintético."
    end

    tr = Array{Float64}(undef, nx, ny)
    tg = Array{Float64}(undef, nx, ny)
    tb = Array{Float64}(undef, nx, ny)

    if img === nothing
        _fill_demo_target!(tr, tg, tb, nx, ny)
        return tr, tg, tb
    end

    H, W = size(img)                      # filas (arriba→abajo), columnas (izq→der)
    @inbounds for x in 1:nx
        c0 = floor(Int, (x - 1) / nx * W) + 1
        c1 = max(c0, floor(Int, x / nx * W))
        for y in 1:ny
            # volteo vertical: y=ny (arriba del plot) ↔ fila 1 (arriba de la imagen)
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

# Patrón de demostración (sin imagen del usuario): un pequeño "paisaje" reconocible
# —cielo en degradado, sol y suelo— para que se aprecie el calcado.
function _fill_demo_target!(tr, tg, tb, nx, ny)
    cx, cy = nx * 0.5, ny * 0.58           # centro del sol
    rad    = min(nx, ny) * 0.20
    @inbounds for x in 1:nx, y in 1:ny
        t = (y - 1) / max(1, ny - 1)       # 0 abajo, 1 arriba
        # cielo: degradado azul
        r = 0.25 + 0.35 * t
        g = 0.45 + 0.35 * t
        b = 0.75 + 0.20 * t
        # suelo (cuarto inferior): verde
        if y < ny * 0.25
            r, g, b = 0.20, 0.55, 0.22
        end
        # sol: círculo naranja
        if (x - cx)^2 + (y - cy)^2 <= rad^2
            r, g, b = 1.0, 0.72, 0.12
        end
        tr[x, y] = r; tg[x, y] = g; tb[x, y] = b
    end
    return nothing
end

# ── post_init: prepara el objetivo y siembra los agentes ────────────────────────
function trace_init!(model)
    props  = abmproperties(model)
    dims   = size(abmspace(model))
    nx, ny = dims
    rng    = abmrng(model)

    path = string(get(props, :image_path, ""))
    tr, tg, tb = _load_or_make_target(path, nx, ny)
    props[:_trace_state] = (tr = tr, tg = tg, tb = tb, nx = nx, ny = ny)

    # Imagen objetivo como fondo opcional del vídeo (Sim 2).
    bg = Matrix{RGB{Float32}}(undef, nx, ny)
    @inbounds for x in 1:nx, y in 1:ny
        bg[x, y] = RGB{Float32}(tr[x, y], tg[x, y], tb[x, y])
    end
    props[:background_image] = bg

    # Siembra: una fracción `density` de celdas, en posiciones distintas, con color aleatorio.
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

# ── Paso del modelo (síncrono) ──────────────────────────────────────────────────
function trace_model_step!(model)
    props    = abmproperties(model)
    st       = props[:_trace_state]
    R        = Int(get(props, :search_radius, 8))
    cham     = Bool(get(props, :chameleon, true))
    rate     = Float64(get(props, :chameleon_rate, 0.15))
    tol2     = Float64(get(props, :arrival_tol, 0.12))^2
    patience = Int(get(props, :patience, 3))
    _trace_run_step!(model, st, R, cham, rate, tol2, patience)   # barrera de tipos (bucle concreto)
end

function _trace_run_step!(model, st, R::Int, cham::Bool, rate::Float64,
                          tol2::Float64, patience::Int)
    tr, tg, tb = st.tr, st.tg, st.tb
    nx, ny     = st.nx, st.ny

    agents = collect(allagents(model))
    n      = length(agents)
    n == 0 && return nothing

    # 1. Cada agente decide (SÍNCRONO: desde el estado actual, nadie se ha movido aún)
    #    su celda destino y si quiere asentarse (está ya en su mejor sitio o suficientemente
    #    cerca). Los ya asentados se quedan.
    desired   = Vector{Tuple{Int,Int}}(undef, n)
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
            desired[i] = (px, py); willsettle[i] = true            # ya en su mejor sitio
        else
            desired[i] = (px + sign(bestx - px), py + sign(besty - py))  # un píxel hacia el objetivo
            willsettle[i] = false
        end
    end

    # 2. Movimiento por PASADAS: en cada pasada, un agente entra en su celda destino solo si
    #    sigue VACÍA (chequeo en vivo ⇒ nunca dos agentes en la misma celda). Varias pasadas
    #    permiten "trenes": un agente sigue a otro que acaba de liberar la celda. El orden es
    #    aleatorio (abmrng) para repartir con justicia los conflictos y ser reproducible.
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

    # 3. Asentamiento (paciencia) + transformación camaleónica (sobre la posición final).
    @inbounds for i in 1:n
        a = agents[i]; s = a.state
        if !s.found
            if willsettle[i]
                s.found = true
            elseif moved[i]
                s.stuck = 0                       # avanzó: reinicia la paciencia
            else
                s.stuck += 1                      # bloqueado: agota paciencia y se asienta
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

# ── Métricas (opcionales, para [run]) ───────────────────────────────────────────
# Error medio de color frente a la imagen objetivo (baja ⇒ mejor calcado).
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

# Fracción de agentes que ya encontraron su sitio.
function found_fraction(model)
    n = nagents(model)
    return n == 0 ? 0.0 : count(a -> a.state.found, allagents(model)) / n
end
