# Pure-logic tests for LLMBuilder (no LLM / network needed).
# Run:  julia --project=. test/test_llmbuilder.jl
include(joinpath(@__DIR__, "..", "src", "LLMBuilder.jl"))
using .LLMBuilder
const L = LLMBuilder

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond; npass += 1; println("PASS: ", name)
    else;    nfail += 1; println("FAIL: ", name); end
end

# 1. load_prompt — common core + category body present
for c in ("grid_discrete", "continuous_field", "continuous_space", "hexagonal")
    p = load_prompt(c)
    check("load_prompt[$c] common core", occursin("Cell_IA", p) && occursin("OUTPUT FORMAT", p))
    check("load_prompt[$c] category body", occursin("CATEGORY:", p))
end

# 2. parse_router_json
r = L.parse_router_json("""noise {"category":"continuous_field","approach":"Lenia","reason":"moves on its own"} tail""")
check("router parse valid", r !== nothing && r.category == "continuous_field")
check("router parse no-json", L.parse_router_json("nope") === nothing)
check("router parse unknown category", L.parse_router_json("""{"category":"foo","approach":"a","reason":"b"}""") === nothing)

# 3. validate_files
gol_toml = """
[simulation]
model_name = "GoL"
seed = 42
[space]
type = "grid"
dimensions = [50, 50]
[agents]
state_type = "Bool"
[population]
pop_density = { "true" = 0.3, "false" = 0.7 }
[rules]
agent_step = "gol_step!"
model_step = "default_model_step!"
[visualization]
filename = "output_videos/gol.mp4"
"""
ok, _ = validate_files(Dict("gol.toml" => gol_toml), "grid_discrete")
check("grid_discrete builtin valid", ok)

fire_toml = replace(gol_toml, "agent_step = \"gol_step!\"" => "agent_step = \"fire_step!\"")
ok, _ = validate_files(Dict("fire.toml" => fire_toml,
    "fire_rules.jl" => "function fire_step!(agent, model)\n model.next_states[agent.id] = agent.state\nend"), "grid_discrete")
check("grid_discrete custom-in-jl valid", ok)

ok, _ = validate_files(Dict("fire.toml" => fire_toml,
    "fire_rules.jl" => "function other!(a, m) end"), "grid_discrete")
check("grid_discrete custom-missing invalid", !ok)

ok, _ = validate_files(Dict("fire.toml" => fire_toml,
    "fire_rules.jl" => "@agent struct X end\nfunction fire_step!(a, m) end"), "grid_discrete")
check("grid_discrete @agent invalid", !ok)

ok, _ = validate_files(Dict("gol.toml" => gol_toml), "continuous_space")
check("space-type mismatch invalid", !ok)

lenia_toml = """
[simulation]
model_name = "Lenia"
seed = 42
[space]
type = "grid"
[agents]
state_type = "Float64"
[population]
pop_density = { "0.0" = 1.0 }
[rules]
model_step = "lenia_model_step!"
post_init = "lenia_init!"
initialization_rule = "uniform_float"
[visualization]
color_scheme = "viridis"
"""
ok, _ = validate_files(Dict("lenia.toml" => lenia_toml), "continuous_field")
check("continuous_field valid", ok)
ok, _ = validate_files(Dict("lenia.toml" => replace(lenia_toml, "\"Float64\"" => "\"Int\"")), "continuous_field")
check("continuous_field wrong state invalid", !ok)

flock_toml = """
[simulation]
model_name = "F"
seed = 42
[space]
type = "continuous"
[agents]
state_type = "Bird"
speed = 1.0
[population]
pop_quantity = { "Bird" = 50 }
[rules]
agent_step = "agent_step!"
[visualization]
agent_shape = "arrow"
"""
flock_jl = "using Agents\n@agent struct Bird(ContinuousAgent{2,Float64})\n const speed::Float64\nend\nfunction agent_step!(b, m) end"
ok, _ = validate_files(Dict("flocking.toml" => flock_toml, "flocking_rules.jl" => flock_jl), "continuous_space")
check("continuous_space valid", ok)
ok, _ = validate_files(Dict("flocking.toml" => flock_toml), "continuous_space")
check("continuous_space missing jl invalid", !ok)

hive_toml = """
[simulation]
model_name = "H"
seed = 42
[space]
type = "hexagonal"
[agents]
state_type = "Bool"
[population]
pop_quantity = { true = 25 }
[rules]
agent_step = "bee_step!"
[visualization]
agent_shape = "hexagon"
"""
ok, _ = validate_files(Dict("hive.toml" => hive_toml, "hive_rules.jl" => "function bee_step!(a, m) end"), "hexagonal")
check("hexagonal valid", ok)
ok, _ = validate_files(Dict("hive.toml" => replace(hive_toml, "\"hexagon\"" => "\"rect\""),
    "hive_rules.jl" => "function bee_step!(a, m) end"), "hexagonal")
check("hexagonal wrong shape invalid", !ok)

println("\n=== $npass passed, $nfail failed ===")
exit(nfail == 0 ? 0 : 1)
