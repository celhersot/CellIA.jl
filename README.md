# 🦠 Cell_IA 1.0🦠
### *A Data-Driven Multi-Agent Systems Framework in Julia*

![Julia](https://img.shields.io/badge/language-Julia-9558b2?style=for-the-badge&logo=julia)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

Welcome to Cell_IA, a modular and highly customizable environment for simulating Agent-Based Models (ABM). This framework allows you to switch between different agent types, space sizes and properties, and evolution rules simply by editing a TOML configuration file without touching the core logic.

---

## 🧬 Core Architecture

The framework is organized into **4 specialized modules**:

1.  **`AgentDefinition`**: 
    * Defines agent data structures (`structs`).
    * Supports only one type called `UniversalAgent` (for discrete logic).
2.  **`EvolutionRules`**:
    * Contains the behavioral logic.
    * Handles **Synchronous** (buffered) and **Asynchronous** evolution.
3.  **`Initialization`**:
    * Sets up the environment (Grids only).
    * Populates the world using strategies like `random_placement`, `sorted_h`, or `sorted_v`.
4.  **`Representation`**:
    * Handles data visualization and video export (`.mp4`).
    * Supports dynamic shapes (Rectangles, Circles, etc) and complex color mappings.

---

## ⚙️ Configuration (TOML)

Everything is **Data-Driven**. The framework reads a `.toml` file to decide how to build and run the simulation.

### Example Configuration:
```toml
[simulation]
model_name = "RockPaperScissors"
seed = 42
steps = 200

[space]
type = "grid"
dimensions = [50, 50]
periodic = true
metric = "chebyshev"

[agents]
n_states = 3
type = "GridAgent"
density = 1.0

[rules]
agent_step = "rps_step_syn!" 
model_step = "gol_model_step!"
initialization_rule = "random_placement"

[properties]
threshold = 3

[visualization]
filename = "rps_simulation.mp4"
title = "Rock Paper Scissors"
agent_shape = "rect"
agent_size = 13
framerate = 15
frames = 50
color_scheme = { 1 = "red", 2 = "green", 3 = "blue" } 
variable_to_color = "state"´
```
---

## 🎮 How to Run
To run the framework, use the main.jl script and pass the path to your desired .toml configuration as an argument.

Available Examples:
Navigate to your project root and execute:

### 1. Game of Life - Synchronous
julia examples/main.jl examples/gol_syn.toml

### 2. Game of Life - Asynchronous
julia examples/main.jl examples/gol_asyn.toml

### 3. Rock-Paper-Scissors
julia examples/main.jl examples/rps_syn.toml

---

## 📂 Project Structure
```plaintext
.
├── src/
│   ├── AgentDefinition.jl          # Agent structs
│   ├── EvolutionRulesDefinition.jl # Logic and Physics
│   ├── InitializationDefinition.jl # Space and Population setup
│   └── RepresentationDefinition.jl # Makie Visualization
├── examples/
│   ├── main.jl                     # Entry point
│   ├── gol_syn.toml                # GoL config (Synchronous)
│   ├── gol_asyn.toml               # GoL config (Asynchronous)
│   └── rps_syn.toml                # Rock-Paper-Scissors config (Synchronous)
└── README.md
```

---

## 🛠️ Customization

* To add a new rule: Define a function in EvolutionRulesDefinition.jl and reference its name in the agent_step field of your TOML.

* To change visuals: Edit the color_scheme or agent_shape in the TOML.

---

Developed by Celia with ❤️ using Agents.jl and CairoMakie.jl.