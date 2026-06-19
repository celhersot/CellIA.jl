using Documenter
using Cell_IA

makedocs(
    sitename = "Cell_IA.jl",
    authors  = "Celia Hermoso Soto",
    modules  = [
        Cell_IA,
        Cell_IA.Initialization,
        Cell_IA.SpaceDefinition,
        Cell_IA.Representation,
        Cell_IA.CustomEvolutionRules,
        Cell_IA.LLMBuilder,
        Cell_IA.UniversalAgents,
        Cell_IA.HexagonalSpace,
    ],
    pages = [
        "Inicio"            => "index.md",
        "Guía de uso"       => "guia.md",
        "Referencia de API" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://celhersot.github.io/Cell_IA.jl",
    ),
    # De momento avisamos (no fallamos) ante docstrings/refs pendientes; se irá endureciendo
    # a medida que se complete la documentación de la API.
    warnonly = true,
)

deploydocs(
    repo      = "github.com/celhersot/Cell_IA",
    devbranch = "main",
)
