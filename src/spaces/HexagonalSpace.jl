module HexagonalSpace

using Agents

export HexagonalGridSpace, HEX_SIZE, hex_to_pixel, hex_corners

const HEX_SIZE = 1.0

# Generic hexagonal grid. Simulation-specific per-cell data goes in cell_properties.
struct HexagonalGridSpace <: Agents.AbstractSpace
    dims::Tuple{Int, Int}
    periodic::Bool
    agent_positions::Dict{Tuple{Int, Int}, Vector{Int}}
    cell_properties::Dict{Tuple{Int, Int}, Dict{Symbol, Any}}
end

function HexagonalGridSpace(dims::Tuple{Int, Int}; periodic::Bool = false)
    agent_positions = Dict{Tuple{Int, Int}, Vector{Int}}()
    cell_properties = Dict{Tuple{Int, Int}, Dict{Symbol, Any}}()
    for q in 1:dims[1], r in 1:dims[2]
        agent_positions[(q, r)] = Int[]
        cell_properties[(q, r)] = Dict{Symbol, Any}()
    end
    return HexagonalGridSpace(dims, periodic, agent_positions, cell_properties)
end

Base.size(space::HexagonalGridSpace) = space.dims

# ── Agents.jl interface ────────────────────────────────────────────────────────

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

# Offset-coordinate neighbours (even/odd row stagger).
function Agents.nearby_positions(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace}, _r=1)
    row, col = pos
    dims     = abmspace(model).dims
    periodic = abmspace(model).periodic

    dirs = row % 2 == 0 ?
        [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)] :
        [(-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1)]

    valid = Tuple{Int, Int}[]
    for (dq, dr) in dirs
        nr, nc = row + dq, col + dr
        if periodic
            push!(valid, (mod1(nr, dims[1]), mod1(nc, dims[2])))
        elseif 1 <= nr <= dims[1] && 1 <= nc <= dims[2]
            push!(valid, (nr, nc))
        end
    end
    return valid
end

function Agents.nearby_ids(pos::Tuple{Int, Int}, model::ABM{<:HexagonalGridSpace}, r::Int=1)
    ids = Int[]
    for p in [pos; Agents.nearby_positions(pos, model, r)]
        append!(ids, abmspace(model).agent_positions[p])
    end
    return ids
end

function Agents.ids_to_inspect(model::ABM{<:HexagonalGridSpace}, pos)
    return abmspace(model).agent_positions[pos]
end

Agents.agents_space_dimensionality(::HexagonalGridSpace) = 2

# ── Geometry helpers (no Makie dependency — return plain float tuples) ─────────

function hex_to_pixel(row::Int, col::Int, size::Float64 = HEX_SIZE)
    offset = (row % 2 == 1) ? 0.5 : 0.0
    x = size * sqrt(3) * (col - 1 + offset)
    y = size * 1.5     * (row - 1)
    return (x, y)
end

function hex_corners(center::Tuple{Float64, Float64}, size::Float64 = HEX_SIZE; gap::Float64 = 0.97)
    cx, cy = center
    [(cx + size * gap * cos(π/6 + π/3*i),
      cy + size * gap * sin(π/6 + π/3*i)) for i in 0:5]
end

end
