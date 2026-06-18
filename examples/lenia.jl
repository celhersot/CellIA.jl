# Reglas opcionales de Lenia. El comportamiento por defecto (kernel y crecimiento
# gaussianos) ya esta en CustomEvolutionRules.jl, asi que lenia.toml funciona sin este
# archivo. Para experimentar, define aqui tu propio lenia_init! fijando :kernel_fn o
# :growth_fn en abmproperties(model) y llamando luego a _lenia_build_kernel!(model).
