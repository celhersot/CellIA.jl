# Lenia rules file — loaded into CustomEvolutionRules via main.jl.
#
# The default behaviour (gaussian kernel + gaussian growth) is already defined
# in CustomEvolutionRules.jl.  Override here to explore different "creatures".
#
# ── How to customise ──────────────────────────────────────────────────────────
#
# Option A — change kernel via abmproperties(model) before the kernel is built:
#
#   function lenia_init!(model)
#       abmproperties(model)[:kernel_fn] = r -> max(0.0, 1.0 - r^2)^4  # polynomial
#       _lenia_build_kernel!(model)                                   # rebuild
#   end
#
# Option B — change growth function (applied every step):
#
#   abmproperties(model)[:growth_fn] = (u, μ, σ) -> begin
#       # double-peaked growth for richer behaviour
#       g1 = exp(-((u - μ)^2)         / (2σ^2))
#       g2 = exp(-((u - μ - 0.1)^2)   / (2σ^2)) * 0.5
#       2(g1 + g2) / 1.5 - 1.0
#   end
#
# Option C — fully custom init (kernel + growth):
#
#   function lenia_init!(model)
#       abmproperties(model)[:kernel_fn] = r -> max(0.0, sin(π * r))
#       abmproperties(model)[:growth_fn] = (u, μ, σ) -> 2exp(-((u-μ)^2)/(2σ^2)) - 1
#       _lenia_build_kernel!(model)
#   end
#
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment one of the blocks above and run:
#   julia examples/main.jl examples/lenia.toml examples/lenia.jl
