# User guide

## Quick start

The most direct way to run a model is the `examples/main.jl` launcher, which reads a TOML and,
optionally, a user-provided rules file:

```bash
# Config only
julia examples/main.jl examples/organismo.toml

# Config + user rules (models that ship their own struct/functions)
julia examples/main.jl examples/lenia_perturbation.toml examples/lenia_perturbation.jl
```

Programmatically the flow is: read the TOML → [`initialize_model`](@ref) → request an output
([`video_simulation`](@ref), [`photo_simulation`](@ref) or [`run_simulation`](@ref)).

## Generating a model from a prompt (local LLM)

[`build_from_prompt`](@ref) turns a natural-language description into a runnable simulation. It
routes the request into one of the four categories, generates the `.toml` (and a `_rules.jl`
when needed), validates the output with a repair loop and runs it:

```julia
using Cell_IA
build_from_prompt("a colony of cells on a honeycomb grid")
```

The first call downloads a ~1 GB model into `models/` and runs it on CPU through
`bin/llama-cli` (set `CELL_IA_GPU_LAYERS` to offload to a GPU). Generated files are written to
`examples/` by default (`output_dir` keyword).

## Where outputs are saved

| Output | Function | Default location |
|---|---|---|
| Video (MP4) | [`video_simulation`](@ref) | `[visualization].filename`, e.g. `output_videos/<name>.mp4` |
| Photos (PNG) | [`photo_simulation`](@ref) | `[visualization].photo_prefix`, default `output_photos/sim*` |
| Metrics (CSV) | [`run_simulation`](@ref) | `[run].output`, default `output_data/results.csv` |

The folders are created automatically and are git-ignored.

## Anatomy of a TOML

A model is described with these sections (not all are mandatory):

| Section | Purpose |
|---|---|
| `[simulation]` | Model name and `seed` (reproducibility). |
| `[space]` | `type` (`grid`/`continuous`/`hexagonal`), `dimensions`, `periodic`, `metric`. |
| `[agents]` | `state_type` (`Bool`, `Int`, `Symbol`, `Float64` or a custom type from the rules). |
| `[population]` | `pop_density` or `pop_quantity` per state: how many agents and of which type. |
| `[properties]` | Global model parameters (e.g. `lenia_mu`, `min_to_live`…). |
| `[rules]` | `initialization_rule`, `agent_step`, `model_step`, `post_init` (names of functions in `CustomEvolutionRules`). |
| `[visualization]` | Video/photo output: `filename`, `frames`, `framerate`, `color_scheme`, `title`. |
| `[run]` | Data collection: `steps`, `output` (CSV) and `mdata`/`adata` (properties or aggregator functions). |

Minimal example (Game of Life):

```toml
[simulation]
model_name = "Game of Life"
seed = 42

[space]
type = "grid"
dimensions = [100, 100]
periodic = true

[agents]
state_type = "Bool"

[population]
pop_density = { "true" = 0.3, "false" = 0.7 }

[properties]
min_to_live = 2
max_to_live = 3

[rules]
initialization_rule = "random"
agent_step = "gol_step!"
model_step = "default_model_step!"

[visualization]
filename = "output_videos/gol.mp4"
frames = 100
framerate = 10
```

## The four simulation categories

Every model falls into one of these four categories, which set the kind of space and rules:

| Category | What it is | Space |
|---|---|---|
| `grid_discrete` | Cellular automaton on a grid where each cell holds a **discrete** state and updates from its neighbours. Game-of-Life style. | `grid` |
| `continuous_field` | **Lenia**-type: a grid of cells holding a **continuous** value in [0, 1] that form an organism flowing across the grid. | `grid` |
| `continuous_space` | **Flocking/boids**: many agents move as points through a continuous space and group with their neighbours. | `continuous` |
| `hexagonal` | Models on a **hexagonal** grid with per-cell properties. Hive style. | `hexagonal` |

## Example catalogue

`examples/` ships ready-to-run configurations:

| Model | TOML | Category / type |
|---|---|---|
| Game of Life | `conway.toml`, `gol.toml` | `grid_discrete` (boolean grid) |
| Rock–Paper–Scissors | `rps.toml` | `grid_discrete` (symbol grid) |
| Schelling segregation | `schelling.toml` | `grid_discrete` (grid with movement) |
| Forest fire | `forestFire.toml` | `grid_discrete` |
| Flocking / boids | `flocking.toml` + `flocking.jl` | `continuous_space` |
| Particle attraction | `particles.toml` + `particles.jl` | `continuous_space` |
| Lenia (orbium) | `lenia.toml`, `organismo.toml` | `continuous_field` (FFT field grid) |
| Lenia under perturbation | `lenia_perturbation.toml` + `lenia_perturbation.jl` | `continuous_field` + stochastic noise & energy metrics |
| Hive | `hive.toml` + `hive.jl` | `hexagonal` |
| Image trace | `image_trace.toml`, `image_trace_pointillist.toml` + `image_trace.jl` | agents reproducing an image |
| Others | `isla.toml` + `isla_rules.jl` | various |

## How to add a new model

1. **Agent state:** use a basic type in `[agents].state_type`, or define your own struct in a
   rules file (like `Painter` in `examples/image_trace.jl`).
2. **Rules:** write the update function(s) (`agent_step!` and/or `model_step!`) and, if you need
   initial setup, a `post_init`. They can live in `CustomEvolutionRules` (generic) or in a user
   file that `main.jl` injects.
3. **Configuration:** create the `.toml` naming those rules under `[rules]`.
4. **Run:** `julia examples/main.jl your_model.toml [your_rules.jl]`.
