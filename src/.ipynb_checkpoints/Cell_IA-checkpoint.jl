module Cell_IA

using Agents, Plots
rules = (2, 3, 3, 3)

mutable struct Cell <: AbstractAgent
    id::Int
    pos::Tuple{Int,Int}
    status::Bool
end

function build_model(; rules::Tuple, dims = (100, 100), Moore = true)
    space = GridSpace(dims, moore = Moore)
    properties = Dict(:rules => rules)
    model = ABM(Cell, space; properties = properties)
    node_idx = 1
    for x in 1:dims[1]
        for y in 1:dims[2]
            add_agent_pos!(Cell(node_idx, (x, y), false), model)
            node_idx += 1
        end
    end
    return model
end

function ca_step!(model)
    new_status = fill(false, nagents(model))
    for (agid, ag) in model.agents
        nlive = nlive_neighbors(ag, model)
        if ag.status == true && (nlive ≤ model.rules[4] && nlive ≥ model.rules[1])
            new_status[agid] = true
        elseif ag.status == false && (nlive ≥ model.rules[3] && nlive ≤ model.rules[4])
            new_status[agid] = true
        end
    end

    for k in keys(model.agents)
        model.agents[k].status = new_status[k]
    end
end

function nlive_neighbors(ag, model)
    neighbors_coords = node_neighbors(ag, model)
    nlive = 0
    for nc in neighbors_coords
        nag = model.agents[Agents.coord2vertex((nc[2], nc[1]), model)]
        if nag.status == true
            nlive += 1
        end
    end
    return nlive
end

model = build_model(rules = rules, dims = (50, 50), Moore = true)

for i in 1:nv(model)
    if rand() < 0.2
        model.agents[i].status = true
    end
end

ac(x) = x.status == true ? :black : :white
anim = @animate for i in 0:100
    i > 0 && step!(model, dummystep, ca_step!, 1)
    p1 = plotabm(model; ac = ac, as = 3, am = :square, showaxis = false)
end

end # module