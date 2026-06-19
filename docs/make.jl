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
        "Home"          => "index.md",
        "User guide"    => "guia.md",
        "API reference" => "api.md",
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
