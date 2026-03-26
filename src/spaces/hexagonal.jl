using Agents
import Agents: add_agent_to_space!, remove_agent_from_space!, 
              random_position, nearby_ids, nearby_positions, space_axis_limits

struct HexagonalSpaceSingle{P} <: Agents.AbstractGridSpace{2, P}
    s::Matrix{Int}
    dims::NTuple{2, Int}
    periodic::Bool
end

function HexagonalSpaceSingle(dims::NTuple{2, Int}; periodic = false)
    s = zeros(Int, dims)
    return HexagonalSpaceSingle{periodic}(s, dims, periodic)
end

function mod_pos(pos, dims)
    return (mod1(pos[1], dims[1]), mod1(pos[2], dims[2]))
end

# 1. nearby_positions: Devuelve las coordenadas de las celdas cercanas
function nearby_positions(pos::NTuple{2, Int}, model::ABM{<:HexagonalSpaceSingle}, r::Real = 1)
    dims = model.space.dims
    periodic = model.space.periodic
    res = NTuple{2, Int}[]
    
    ri = Int(floor(r))
    # Iteramos en el "bounding box" del radio r
    for dq in -ri:ri
        for dr in max(-ri, -dq-ri):min(ri, -dq+ri)
            dq == dr == 0 && continue
            new_pos = (pos[1] + dq, pos[2] + dr)
            
            if periodic
                push!(res, mod_pos(new_pos, dims))
            else
                if 1 <= new_pos[1] <= dims[1] && 1 <= new_pos[2] <= dims[2]
                    push!(res, new_pos)
                end
            end
        end
    end
    return unique(res)
end

# 2. nearby_ids: Aprovecha nearby_positions para sacar los IDs
function nearby_ids(pos::NTuple{2, Int}, model::ABM{<:HexagonalSpaceSingle}, r::Real = 1)
    positions = nearby_positions(pos, model, r)
    ids = Int[]
    for p in positions
        id = model.space.s[p...]
        id != 0 && push!(ids, id)
    end
    return ids
end

function Agents.add_agent_to_space!(agent, model::ABM{<:HexagonalSpaceSingle})
    pos = agent.pos
    # Si la celda está ocupada (ID != 0), lanzamos error en SpaceSingle
    if model.space.s[pos...] != 0
        error("¡Choque! La celda hexagonal $(pos) ya está ocupada por el agente $(model.space.s[pos...])")
    end
    model.space.s[pos...] = agent.id
    return agent
end

# Para sacar al agente (limpiar la celda)
function Agents.remove_agent_from_space!(agent, model::ABM{<:HexagonalSpaceSingle})
    model.space.s[agent.pos...] = 0
    return agent
end

# Extra: Mover agentes es automático, pero por eficiencia puedes definirlo:
function Agents.move_agent!(agent, pos::NTuple{2, Int}, model::ABM{<:HexagonalSpaceSingle})
    # Quitamos de la posición vieja
    model.space.s[agent.pos...] = 0
    # Ponemos en la nueva
    agent.pos = pos
    model.space.s[pos...] = agent.id
    return agent
end