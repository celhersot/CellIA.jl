module InitializationDefinition

export initialize_model

include("../prueba/AgentDefinition.jl")
include("../prueba/EvolutionRulesDefinition.jl")

using Agents
using Random
using .AgentDefinition # Asumiendo que usas '..' para subir un nivel
using .EvolutionRulesDefinition

function initialize_model(config::Dict)
    # 1. Definir el Espacio
    space_config = config["space"]
    space = nothing
    
    if space_config["type"] == "grid"
        dims = Tuple(space_config["dimensions"])
        # GridSpaceSingle es más rápido si solo hay 1 agente por celda
        space = GridSpaceSingle(dims; periodic = space_config["periodic"])
    elseif space_config["type"] == "continuous"
        # Lógica para ContinuousSpace...
        space = ContinuousSpace(Tuple(space_config["dimensions"]))
    end

    # 2. Definir Propiedades del Modelo (Globales)
    # Convertimos las keys de String a Symbol para Agents.jl
    properties = Dict(Symbol(k) => v for (k, v) in config["properties"])

    # 3. Obtener la Regla de Evolución dinámicamente
    rule_name = config["rules"]["evolution_rule"]
    # MAGIA: Buscamos la función dentro del módulo EvolutionRules usando su nombre
    step_function = getfield(EvolutionRulesDefinition, Symbol(rule_name))

    # 4. Construir el Modelo
    rng = Xoshiro(config["simulation"]["seed"])
    
    # Aquí asumimos UniversalAgent por simplicidad, pero podrías elegir el struct también
    model = StandardABM(
        UniversalAgent, 
        space;
        agent_step! = step_function,
        properties = properties,
        rng = rng,
        scheduler = Schedulers.Randomly()
    )

    # 5. Inicializar Agentes (Poblar el mundo)
    populate_world!(model, config)

    return model
end

function populate_world!(model, config)
    # Lógica genérica para llenar el mundo según densidad o número total
    # Esto también podría ser una función dinámica si la inicialización es compleja
    agent_cfg = config["agents"]
    total_agents = 0
    
    if haskey(agent_cfg, "total_count") && agent_cfg["total_count"] > 0
        total_agents = agent_cfg["total_count"]
    else
        dims = size(abmspace(model))
        total_slots = prod(dims)
        total_agents = floor(Int, total_slots * agent_cfg["density"])
    end

    for _ in 1:total_agents
        # Estado inicial por defecto (puedes parametrizarlo más)
        add_agent_single!(model; state = true, group = 1) 
    end
end

end