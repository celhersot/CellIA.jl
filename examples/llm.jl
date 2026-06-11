# Entry point for the local-LLM simulation generator.
#
#   julia --project=. examples/llm.jl "describe tu simulación"
#   julia --project=. examples/llm.jl "..." --yes      # no confirmación (usa la 1ª propuesta)

include("../src/LLMBuilder.jl")
using .LLMBuilder

if abspath(PROGRAM_FILE) == @__FILE__
    args = collect(ARGS)
    yes  = "--yes" in args
    filter!(a -> a != "--yes", args)

    if length(args) >= 1
        description = join(args, " ")
        build_from_prompt(description; interactive = !yes)
    else
        println("Uso: julia --project=. examples/llm.jl \"describe tu simulación\" [--yes]")
    end
end
