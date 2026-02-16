import Pkg

using TOML

include("../src/InitializationDefinition.jl")
include("../src/RepresentationDefinition.jl")

using .InitializationDefinition
using .RepresentationDefinition

function main(config_file::String="gol_syn.toml")
    # Is conif_file existing?
    if !isfile(config_file)
        error("The configuration file '$config_file' does not exist.")
    end

    println("--> Extracting the configuration from: $config_file")
    config = TOML.parsefile(config_file)

    sim_name = get(config["simulation"], "model_name", "Unknown Simulation")
    println("--> Detected model: $sim_name")

    println("--> Initializing model and agentes...")
    model = initialize_model(config)

    println("--> Starting simulation and generating video...")
    viz_config = config["visualization"]
    
    video_simulation(model, viz_config)

    println("--> Finished! Simulation saved in: $(viz_config["filename"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    # It allows exectue from cmd: julia main.jl config_file.toml
    if length(ARGS) > 0
        main(ARGS[1])
    else
        main() # Uses "gol_syn.toml" by default
    end
end