module Cell_IA

include("UniversalAgents.jl")
include("SpaceDefinition.jl")
include("CustomEvolutionRules.jl")
include("Initialization.jl")
include("LLMBuilder.jl")
include("Representation.jl")

using .UniversalAgents
using .SpaceDefinition
using .CustomEvolutionRules
using .Initialization
using .LLMBuilder
using .Representation

export build_from_prompt
end