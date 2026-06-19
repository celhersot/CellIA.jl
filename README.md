# Cell_IA

*A data-driven Agent-Based Modelling (ABM) framework in Julia.*

[![CI](https://github.com/celhersot/Cell_IA.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/celhersot/Cell_IA.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Cell_IA lets you define and run agent-based simulations entirely from a TOML
configuration file, without modifying the source code. It builds the model with
[Agents.jl](https://github.com/JuliaDynamics/Agents.jl) and renders the result as an
MP4 video (or PNG photos / a CSV of metrics) with [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl).

## Installation

Until the package is registered in the Julia General registry, install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/celhersot/Cell_IA.jl")
```

## Running a simulation

```bash
# Config only (built-in rules)
julia examples/main.jl examples/gol.toml

# Config + a custom rules file
julia examples/main.jl examples/rps.toml examples/rules.jl
```

Outputs are written to `output_videos/` (and `output_photos/` / `output_data/` when a
`[visualization]` photo or `[run]` block is configured).

### Example models

| Config | Rules file | Description |
|--------|-----------|-------------|
| `gol.toml` | built-in | Conway's Game of Life |
| `rps.toml` | built-in | Rock–Paper–Scissors |
| `schelling.toml` | built-in | Schelling segregation |
| `conway.toml` | `conway_rules.jl` | Game of Life variant |
| `forestFire.toml` | `rules.jl` | Forest fire with struct states |
| `isla.toml` | `isla_rules.jl` | Tree/water spread |
| `flocking.toml` | `flocking.jl` | Boids in continuous space |
| `particles.toml` | `particles.jl` | Particle attraction (continuous) |
| `hive.toml` | `hive.jl` | Bees on a hexagonal grid |
| `lenia.toml` | built-in | Lenia continuous cellular automaton |
| `lenia_perturbation.toml` | `lenia_perturbation.jl` | Lenia under stochastic noise |
| `organismo.toml` | built-in | Lenia "orbium" creature |
| `image_trace.toml` | `image_trace.jl` | Colour agents that reproduce an image |
| `image_trace_pointillist.toml` | `image_trace.jl` | Pointillist variant over the target image |

## Architecture

The package (`src/Cell_IA.jl`) is split into modules:

| Module | Role |
|--------|------|
| `UniversalAgents` | Generic `UniversalAgent{T}` agent (`T` = `Bool`, `Int`, `Float64`, `Symbol`, or a custom struct) |
| `SpaceDefinition` | Builds the space (`grid`, `continuous`, `hexagonal`) from the config |
| `HexagonalSpace` | Hexagonal grid space implementation |
| `CustomEvolutionRules` | Built-in rules (Game of Life, RPS, Schelling, Lenia…) and lattice metrics |
| `Initialization` | Reads the config, resolves types and rule names, builds the model and populates it |
| `Representation` | Video, photo and CSV export |
| `LLMBuilder` | Experimental local-LLM generator of simulations (optional) |

## Configuration (TOML)

```toml
[simulation]    # model_name, seed
[space]         # type ("grid"/"continuous"/"hexagonal"), dimensions, periodic, metric
[agents]        # state_type, initial state fields
[population]    # pop_density or pop_quantity (per state)
[properties]    # model-level parameters (thresholds, rates, Lenia μ/σ…)
[rules]         # agent_step, model_step, initialization_rule, post_init (function names)
[visualization] # filename, framerate, frames, color_scheme, agent_shape…
[run]           # optional: steps, output CSV, adata/mdata metrics
```

`color_scheme` accepts a named colormap (`"viridis"`), a per-state `Dict`, or `"rgb"`
(each agent paints its own cell with an RGB carried in its state).

## Adding a new model

1. Write a TOML config in `examples/`.
2. If the built-in rules are not enough, write a `.jl` file with the rule functions
   (`rule!(agent, model)` / `rule!(model)`) and any custom state struct.
3. Reference function and type names as strings in the TOML.
4. Run it with `julia examples/main.jl examples/yourmodel.toml examples/yourrules.jl`.

## Tests

```julia
using Pkg
Pkg.test("Cell_IA")
```

## License

MIT — see [LICENSE](LICENSE).
