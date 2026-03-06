using Agents
using Random

struct TreeState
    status::Symbol  # :tree, :fire, :empty, :ash
    energy::Int
    age::Int
end

function TreeState(s::String)
    # Limpiamos el string "TreeState(tree, 10, 0)" y sacamos los valores
    parts = split(replace(s, r"TreeState\(|\)" => ""), ",")
    status = Symbol(strip(parts[1]))
    energy = parse(Int, strip(parts[2]))
    age = parse(Int, strip(parts[3]))
    return TreeState(status, energy, age)
end

# 2. La Regla del Forest Fire con Structs
function forest_step!(agent, model)
    current = agent.state
    
    # Lógica: Si está ardiendo, muere y deja ceniza (pierde energía)
    if current.status == :fire
        model.next_states[agent.id] = TreeState(:ash, 0, current.age)
    
    # Si es un árbol, puede prenderse si hay fuego cerca
    elseif current.status == :tree
        # Contamos vecinos que están en estado :fire
        fire_neighbors = count(n -> n.state.status == :fire, nearby_agents(agent, model))
        
        if fire_neighbors > 0 || rand() < model.lightning_chances
            # Se convierte en fuego, mantiene su energía actual
            model.next_states[agent.id] = TreeState(:fire, current.energy, current.age + 1)
        else
            # Sigue siendo árbol, quizás gana un poco de energía/edad
            model.next_states[agent.id] = TreeState(:tree, current.energy + 1, current.age + 1)
        end
    else
        # Si es ceniza o vacío, se queda igual (o recupera energía muy lento)
        model.next_states[agent.id] = current
    end
end