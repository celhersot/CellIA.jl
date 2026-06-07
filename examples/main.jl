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

# Using user code
function load_user_rules(rules_file::String)
    if !isfile(rules_file)
        error("Archivo de reglas no encontrado: $rules_file")
    end
    println("--> Inyectando reglas de usuario desde: $rules_file")
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
        println("--> No se proporcionó archivo de reglas. Usando reglas predefinidas.")
    end

    Base.invokelatest(_execute, config)
    println("--> Done.")
end

function _execute(config)
    name = config["simulation"]["model_name"]
    viz      = get(config, "visualization", nothing)
    run_conf = get(config, "run", nothing)

    # Each output gets its own freshly-initialized model so they are independent
    # (run! and video would otherwise advance the same model cumulatively).
    if !isnothing(run_conf)
        println("--> Initializing $name (run)...")
        model = initialize_model(config)
        run_simulation(model, run_conf,
                       isnothing(viz) ? Dict{String,Any}() : viz,
                       config["space"])
    end

    if !isnothing(viz) && haskey(viz, "filename")
        println("--> Initializing $name (video)...")
        model = initialize_model(config)
        println("--> Generating video...")
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