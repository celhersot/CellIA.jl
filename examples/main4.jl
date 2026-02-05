import Pkg
# Pkg.activate(".") # Descomentar si usas un entorno local
using TOML

# --- CARGA DE MÓDULOS ---
# Ajusta las rutas según tu estructura de carpetas (ej. "../src/" o "./src/")
include("../prueba/AgentDefinition.jl")
include("../prueba/EvolutionRulesDefinition.jl")
include("../prueba/InitializationDefinition.jl")
include("../prueba/RepresentationDefinition.jl")

using .InitializationDefinition
using .RepresentationDefinition

# --- FUNCIÓN PRINCIPAL ---
function main(config_file::String="config.toml")
    # 1. Validar existencia del archivo
    if !isfile(config_file)
        error("El archivo de configuración '$config_file' no existe.")
    end

    println("--> Leyendo configuración desde: $config_file")
    config = TOML.parsefile(config_file)

    # Mostrar info básica para feedback al usuario
    sim_name = get(config["simulation"], "model_name", "Simulación Desconocida")
    println("--> Modelo detectado: $sim_name")

    # 2. INICIALIZACIÓN (Factory Pattern)
    # El módulo Initialization se encarga de interpretar las reglas, espacios y agentes
    println("--> Inicializando modelo y agentes...")
    model = initialize_model(config)

    # 3. REPRESENTACIÓN Y EJECUCIÓN
    # Pasamos el modelo y la configuración visual al módulo de representación
    println("--> Comenzando simulación y generación de video...")
    viz_config = config["visualization"]
    
    # Esta función se encargará de correr la simulación paso a paso mientras graba
    video_simulation(model, viz_config)

    println("--> ¡Listo! Simulación guardada en: $(viz_config["filename"])")
end

# --- PUNTO DE ENTRADA ---
if abspath(PROGRAM_FILE) == @__FILE__
    # Permite ejecutar desde terminal: julia main.jl mi_config_lenia.toml
    if length(ARGS) > 0
        main(ARGS[1])
    else
        main() # Usa "config.toml" por defecto
    end
end