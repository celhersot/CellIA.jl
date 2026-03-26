using Agents
using Random
import Makie
using Makie
using CairoMakie

# Paso 3 STRUCT
struct HexagonalGridSpace <: Agents.AbstractSpace
    dims::Tuple{Int, Int}
    agent_positions::Dict{Tuple{Int, Int}, Vector{Int}} # Dict{Posición, IDs de agentes}
end

function HexagonalGridSpace(dims::Tuple{Int, Int})
    positions = Dict{Tuple{Int, Int}, Vector{Int}}()
    for q in 1:dims[1], r in 1:dims[2]
        positions[(q, r)] = Int[]
    end
    return HexagonalGridSpace(dims, positions)
end
# Las cosas obligatorias de la API

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

# # Devuelve las posiciones vecinas en un grid hexagonal
function Agents.nearby_positions(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace})
    row, col = pos
    # Los 6 vecinos
    directions_even = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1)]
    directions_odd = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1)]
    
    # Calculo posiciones y filtro para que no se salgan del tablero
    valid_positions = Tuple{Int, Int}[]
    directions = []

    if row % 2 == 0
        directions = directions_even
    else
        directions = directions_odd
    end

    for (dq, dr) in directions
        new_row, new_col = row + dq, col + dr
        if 1 <= new_row <= abmspace(model).dims[1] && 1 <= new_col <= abmspace(model).dims[2]
            push!(valid_positions, (new_row, new_col))
        end
    end
    return valid_positions
end

function Agents.nearby_ids(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace}, r::Int = 1)
    positions_to_check = [pos; Agents.nearby_positions(pos, model, r)]
    
    ids = Int[]
    for p in positions_to_check
        append!(ids, abmspace(model).agent_positions[p])
    end
    return ids
end

# -------------------
@agent struct Bee(GridAgent{2})
    honey::Float64
end

function bee_step!(bee::Bee, model)
    vecinos = nearby_positions(bee.pos, model)
    if !isempty(vecinos)
        nueva_pos = rand(abmrng(model), vecinos)
        move_agent!(bee, nueva_pos, model)
    end
    bee.honey += 0.5
end

function initialize_hive(; dims=(10, 10), num_bees=15)
    space = HexagonalGridSpace(dims)
    # 2. Creamos el modelo
    model = StandardABM(Bee, space; agent_step! = bee_step!)
    
    for _ in 1:num_bees
        add_agent!(model; honey = 0.0) 
    end
    
    return model
end

# model = initialize_hive()
# println("Estado inicial: ", model)

# step!(model, 5)
# println("Estado tras 5 pasos: ", model)
# println("Miel de la abeja 1: ", model[1].honey)


# ---------------------------------------- VISUALIZACIÓN
