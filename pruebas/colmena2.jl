using Agents
using Random
using CairoMakie
#using GeometryBasics # Necesario para procesar las formas poligonales

# ============================================================
# ESPACIO HEXAGONAL PERSONALIZADO
# ============================================================

struct HexagonalGridSpace <: Agents.AbstractSpace
    dims::Tuple{Int, Int}
    agent_positions::Dict{Tuple{Int, Int}, Vector{Int}}
    # ¡NUEVO! Ahora la colmena guarda la cantidad de miel por celda
    honey::Dict{Tuple{Int, Int}, Float64} 
end

function HexagonalGridSpace(dims::Tuple{Int, Int})
    positions = Dict{Tuple{Int, Int}, Vector{Int}}()
    honey = Dict{Tuple{Int, Int}, Float64}()
    for q in 1:dims[1], r in 1:dims[2]
        positions[(q, r)] = Int[]
        honey[(q, r)] = 0.0 # Inicializamos todas las celdas vacías de miel
    end
    return HexagonalGridSpace(dims, positions, honey)
end

function Agents.random_position(model::ABM{<:HexagonalGridSpace})
    q = rand(abmrng(model), 1:abmspace(model).dims[1])
    r = rand(abmrng(model), 1:abmspace(model).dims[2])
    return (q, r)
end

function Agents.add_agent_to_space!(agent, model::ABM{<:HexagonalGridSpace})
    push!(abmspace(model).agent_positions[agent.pos], agent.id)
    return agent
end

function Agents.remove_agent_from_space!(agent, model::ABM{<:HexagonalGridSpace})
    filter!(id -> id != agent.id, abmspace(model).agent_positions[agent.pos])
    return agent
end

function Agents.nearby_positions(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace}, r=1)
    row, col = pos
    # ¡CORREGIDO! La lógica de vecinos estaba invertida. Ahora respeta la forma de la malla.
    if row % 2 == 0
        directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    end

    valid_positions = Tuple{Int, Int}[]
    for (dq, dr) in directions
        new_row, new_col = row + dq, col + dr
        if 1 <= new_row <= abmspace(model).dims[1] && 1 <= new_col <= abmspace(model).dims[2]
            push!(valid_positions, (new_row, new_col))
        end
    end
    return valid_positions
end

function Agents.nearby_ids(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace}, r::Int=1)
    positions_to_check = [pos; Agents.nearby_positions(pos, model, r)]
    ids = Int[]
    for p in positions_to_check
        append!(ids, abmspace(model).agent_positions[p])
    end
    return ids
end

# ============================================================
# AGENTE Y LÓGICA DE SIMULACIÓN
# ============================================================

# La abeja ya no tiene la variable "honey" porque ahora la miel está en la celda
@agent struct Bee(GridAgent{2})
end

function bee_step!(bee::Bee, model)
    # 1. Moverse
    vecinos = nearby_positions(bee.pos, model)
    if !isempty(vecinos)
        nueva_pos = rand(abmrng(model), vecinos)
        move_agent!(bee, nueva_pos, model)
    end
    
    # 2. Depositar miel en la celda donde ha caído
    abmspace(model).honey[bee.pos] += 0.5
end

function initialize_hive(; dims=(12, 14), num_bees=25, seed=42)
    space = HexagonalGridSpace(dims)
    model = StandardABM(Bee, space; agent_step! = bee_step!, rng=MersenneTwister(seed))
    for _ in 1:num_bees
        add_agent!(model)
    end
    return model
end

# ============================================================
# GEOMETRÍA HEXAGONAL (AUXILIARES)
# ============================================================

const HEX_SIZE = 1.0

function hex_to_pixel(row::Int, col::Int, size::Float64=HEX_SIZE)
    offset = (row % 2 == 1) ? 0.5 : 0.0
    x = size * sqrt(3) * (col - 1 + offset)
    y = size * 1.5     * (row - 1)
    return Point2f(x, y)
end

function hex_corners(center::Point2f, size::Float64=HEX_SIZE; gap::Float64=0.96) # Gap algo mayor para juntarlos más
    [center + Point2f(size * gap * cos(π/6 + π/3 * i),
                      size * gap * sin(π/6 + π/3 * i)) for i in 0:5]
end

# ============================================================
# API DE VISUALIZACIÓN
# ============================================================

const ABMPlot = Agents.get_ABMPlot_type()

Agents.agents_space_dimensionality(space::HexagonalGridSpace) = 2

function Agents.get_axis_limits(model::ABM{<:HexagonalGridSpace})
    rows, cols = abmspace(model).dims
    s = Float64(HEX_SIZE)
    m = s * 1.2
    xmin = 0.0 - m
    xmax = s * sqrt(3) * (cols - 0.5) + m
    ymin = 0.0 - m
    ymax = s * 1.5 * (rows - 1) + m
    return (xmin, ymin), (xmax, ymax) 
end

function Agents.agentsplot!(ax, p::ABMPlot)
    model_obs = p.abmobs[].model
    pos = @lift [hex_to_pixel(a.pos[1], a.pos[2]) for a in allagents($model_obs)]

    # Cambiamos el marcador a :hexagon y el color a negro
    scatter!(ax, pos;
             color       = :black,
             markersize  = 16,        # Un poco más pequeñas para que quepan en la celda
             marker      = :hexagon, 
             strokecolor = :white,    # Borde blanco para que resalten
             strokewidth = 1)
end

function Agents.spaceplot!(ax, p::ABMPlot; spaceplotkwargs...)
    model_obs = p.abmobs[].model
    dims      = abmspace(model_obs[]).dims

    for row in 1:dims[1], col in 1:dims[2]
        center  = hex_to_pixel(row, col)
        corners = hex_corners(center)

        cell_color = @lift begin
            # Leemos la miel de ESTA celda
            miel = abmspace($model_obs).honey[(row, col)]
            
            if miel > 0
                # Se vuelve más amarillo/anaranjado cuanta más miel tiene
                intensidad = clamp(miel / 15.0, 0.0, 1.0)
                return RGBAf(1.0, 1.0 - 0.2*intensidad, 0.8 - 0.8*intensidad, 1.0)
            else
                # Celdas vacías son blancas
                return :white 
            end
        end

        # Dibujamos el polígono con borde gris oscuro
        poly!(ax, corners;
              color       = cell_color,
              strokecolor = :darkgrey,
              strokewidth = 1.5)
    end
end

function Agents.convert_element_pos(::HexagonalGridSpace, pos)
    row, col = pos
    return hex_to_pixel(row, col)
end

function Agents.ids_to_inspect(model::ABM{<:HexagonalGridSpace}, pos)
    return abmspace(model).agent_positions[pos]
end

# ============================================================
# EJECUCIÓN DEL VÍDEO
# ============================================================

model = initialize_hive(dims=(12, 14), num_bees=25)

abmvideo(
    "colmena2.mp4", model;
    frames      = 80,
    framerate   = 10,
    figure      = (size=(700, 600), backgroundcolor = :white), # Fondo de la imagen blanco
    axis        = (title      = "Colmena en funcionamiento",
                   titlealign = :center,
                   titlecolor = :black,
                   titlesize  = 20,
                   aspect     = DataAspect()),
    enable_space_checks = true
)

println("✓ Vídeo guardado: colmena.mp4")