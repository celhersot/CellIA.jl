include("../src/spaces/HexagonalSpace.jl")
include("../src/UniversalAgents.jl")
include("../src/CustomEvolutionRules.jl")
include("../src/SpaceDefinition.jl")
include("../src/Initialization.jl")
include("../src/Representation.jl")
include("../src/LLMBuilder.jl")

using .LLMBuilder

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        description = join(ARGS, " ")
        build_from_prompt(description)
    else
        println("Use: julia --project=. examples/llm.jl \"describe your simulation here\"")
    end
end
