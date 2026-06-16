# Lanzador del LLMBuilder de Cell_IA.
#
# Uso:
#   julia --project=. examples/llm_build.jl "descripción en lenguaje natural"
#
# Opciones:
#   --no-interactive   elige la categoría automáticamente sin preguntar
#   --out DIR          directorio de salida (por defecto examples/llm_out/)

include(joinpath(@__DIR__, "..", "src", "Cell_IA.jl"))
using .Cell_IA.LLMBuilder

function main(args)
    isempty(args) && error("Uso: julia --project=. examples/llm_build.jl \"descripción\"")

    interactive = true
    out_dir = joinpath(@__DIR__, "llm_out")
    desc_parts = String[]

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--no-interactive"
            interactive = false
        elseif a == "--out" && i < length(args)
            i += 1
            out_dir = args[i]
        else
            push!(desc_parts, a)
        end
        i += 1
    end

    isempty(desc_parts) && error("Falta la descripción de la simulación.")
    build_from_prompt(join(desc_parts, " "); interactive = interactive, output_dir = out_dir)
end

main(ARGS)
