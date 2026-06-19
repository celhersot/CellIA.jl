import Pkg

using TOML

include("../src/spaces/HexagonalSpace.jl")
include("../src/UniversalAgents.jl")
include("../src/CustomEvolutionRules.jl")
include("../src/SpaceDefinition.jl")
include("../src/Initialization.jl")
include("../src/Representation.jl")

using .Initialization
using .Representation

# Carga un archivo de reglas de usuario dentro de CustomEvolutionRules.
function load_user_rules(rules_file::String)
    if !isfile(rules_file)
        error("Archivo de reglas no encontrado: $rules_file")
    end
    println("Cargando reglas de usuario: $rules_file")
    Base.include(Main.CustomEvolutionRules, abspath(rules_file))
end

function main(config_file::String, rules_file::Union{String, Nothing}=nothing)
    # 1. Configuration
    if !isfile(config_file)
        error("Config file not found: $config_file")
    end
    config = TOML.parsefile(config_file)

    # 2. Rules
    if !isnothing(rules_file) && isfile(rules_file)
        load_user_rules(rules_file)
    else
        println("Sin archivo de reglas; uso las predefinidas.")
    end

    Base.invokelatest(_execute, config)
    println("Listo.")
end

function _execute(config)
    name = config["simulation"]["model_name"]
    viz      = get(config, "visualization", nothing)
    run_conf = get(config, "run", nothing)

    # Cada salida usa su propio modelo recien inicializado (run! y video son independientes).
    if !isnothing(run_conf)
        println("Inicializando $name (run)...")
        model = initialize_model(config)
        run_simulation(model, run_conf,
                       isnothing(viz) ? Dict{String,Any}() : viz,
                       config["space"])
    end

    if !isnothing(viz) && haskey(viz, "filename")
        println("Inicializando $name (video)...")
        model = initialize_model(config)
        println("Generando video...")
        video_simulation(model, viz, config["space"])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1
        main(ARGS[1])                # Only config.toml
    elseif length(ARGS) >= 2
        main(ARGS[1], ARGS[2])       # Also rules.jl
    else
        println("Use: julia main.jl <config.toml> [rules.jl]")
    end
end