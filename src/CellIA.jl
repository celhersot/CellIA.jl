module CellIA

include("spaces/HexagonalSpace.jl")
include("UniversalAgents.jl")
include("SpaceDefinition.jl")
include("CustomEvolutionRules.jl")
include("Initialization.jl")
include("LLMBuilder.jl")
include("Representation.jl")

using .HexagonalSpace
using .UniversalAgents
using .SpaceDefinition
using .CustomEvolutionRules
using .Initialization
using .LLMBuilder
using .Representation

export build_from_prompt, HexagonalGridSpace,
       initialize_model, create_space,
       run_simulation, video_simulation, photo_simulation
end