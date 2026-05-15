import Pkg

using TOML

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

    Base.invokelatest(run_simulation, config)
    println("--> Done. Output in: $(config["visualization"]["filename"])")
end

function run_simulation(config)
     # 3. Initialization
    println("--> Initializing $(config["simulation"]["model_name"])...")
    model = initialize_model(config)
    println("--> Generating video...")
    # 4. Visualization
    photo_simulation(model, config["visualization"], config["space"], 7)
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