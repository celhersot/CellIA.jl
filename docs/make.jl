using Documenter
using CellIA

makedocs(
    sitename = "CellIA.jl",
    authors  = "Celia Hermoso Soto",
    modules  = [
        CellIA,
        CellIA.Initialization,
        CellIA.SpaceDefinition,
        CellIA.Representation,
        CellIA.CustomEvolutionRules,
        CellIA.LLMBuilder,
        CellIA.UniversalAgents,
        CellIA.HexagonalSpace,
    ],
    pages = [
        "Inicio"            => "index.md",
        "Guía de uso"       => "guia.md",
        "Referencia de API" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://celhersot.github.io/CellIA.jl",
    ),
    warnonly = true,
)

deploydocs(
    repo      = "github.com/celhersot/CellIA.jl",
    devbranch = "main",
)