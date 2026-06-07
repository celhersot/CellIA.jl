# Entry point for the local-LLM simulation generator.
# Requires llama-server running (see bin/start_server.ps1).
#
#   julia --project=. examples/llm.jl "describe tu simulación"
#   julia --project=. examples/llm.jl "..." --yes      # no confirmación (usa la 1ª propuesta)
#
# LLMBuilder is standalone: it routes, generates and then runs the simulation in a
# subprocess (examples/main.jl), so the framework modules don't need to be loaded here.

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
        println("Antes, arranca el modelo:  pwsh bin/start_server.ps1")
    end
end
