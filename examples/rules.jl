using Agents
using Random

struct TreeState
    status::Symbol  # :tree, :fire, :empty, :ash
    energy::Int
    age::Int
end

function TreeState(s::String)
    parts = split(replace(s, r"TreeState\(|\)" => ""), ",")
    status = Symbol(strip(parts[1]))
    energy = parse(Int, strip(parts[2]))
    age = parse(Int, strip(parts[3]))
    return TreeState(status, energy, age)
end

function forest_step!(agent, model)
    current = agent.state
    
    if current.status == :fire
        model.next_states[agent.id] = TreeState(:ash, 0, current.age)
    
    elseif current.status == :tree
        fire_neighbors = count(n -> n.state.status == :fire, nearby_agents(agent, model))
        
        if fire_neighbors > 0 || rand() < model.lightning_chances
            model.next_states[agent.id] = TreeState(:fire, current.energy, current.age + 1)
        else
            model.next_states[agent.id] = TreeState(:tree, current.energy + 1, current.age + 1)
        end
    else
        model.next_states[agent.id] = current
    end
end