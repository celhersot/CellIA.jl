import Pkg

# Cargar dependencias del proyecto
# Pkg.activate(".") # Descomentar si no lo lanzas con --project

using TOML

# --- 1. Definición de módulos Core ---
# Incluimos los archivos "base" pero NO los usamos todavía
include("../src/UniversalAgents.jl")
include("../src/CustomEvolutionRules.jl") # Módulo vacío
include("../src/SpaceDefinition.jl")

# --- 2. Inyección de Código de Usuario ---
function load_user_rules(rules_file::String)
    if !isfile(rules_file)
        error("Archivo de reglas no encontrado: $rules_file")
    end
    println("--> Inyectando reglas de usuario desde: $rules_file")
    
    # MAGIA: Evaluamos el archivo dentro del módulo CustomEvolutionRules
    # Esto hace que las funciones definidas en reglas.jl pertenezcan a CustomEvolutionRules
    Base.include(Main.CustomEvolutionRules, abspath(rules_file))
end

# --- 3. Cargar resto de módulos dependientes ---
# Initialization depende de que CustomEvolutionRules ya tenga las funciones
include("../src/Initialization.jl")
include("../src/Representation.jl")

using .Initialization
using .Representation

function main(config_file::String, rules_file::Union{String, Nothing}=nothing)
    # 1. Configuración
    if !isfile(config_file)
        error("Config file not found: $config_file")
    end
    config = TOML.parsefile(config_file)

    # 2. Reglas
    if !isnothing(rules_file) && isfile(rules_file)
        load_user_rules(rules_file)
    else
        println("--> No se proporcionó archivo de reglas. Usando reglas predefinidas.")
    end

    Base.invokelatest(run_simulation, config)
    println("--> Done. Output in: $(config["visualization"]["filename"])")
end

function run_simulation(config)
     # 3. Inicializar
    println("--> Initializing $(config["simulation"]["model_name"])...")
    model = initialize_model(config)
    println("--> Generating video...")
    # 4. Visualizar
    video_simulation(model, config["visualization"])
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