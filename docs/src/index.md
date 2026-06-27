# CellIA.jl

*Framework de simulación de autómatas y modelos basados en agentes (ABM) configurables por TOML.*

CellIA es un framework escrito en Julia, construido sobre
[Agents.jl](https://github.com/JuliaDynamics/Agents.jl), cuyo objetivo es **definir y
ejecutar simulaciones sin escribir código**: el usuario describe el modelo en un archivo
`.toml` (espacio, agentes, población, reglas, visualización) y el framework lo construye,
lo ejecuta y genera la salida (vídeo MP4, fotos PNG o datos CSV).

## Filosofía

- **Configuración, no programación.** Un modelo es un TOML; el motor es genérico y no conoce
  los modelos concretos.
- **Reglas modulares.** Las dinámicas viven en `CustomEvolutionRules`; añadir un modelo nuevo
  es escribir una función de regla y, si hace falta, un struct de estado.
- **Mismo motor, muchas salidas.** A partir de un modelo se puede pedir un vídeo, una foto del
  estado, o una recogida de métricas a CSV.
- **Diseño asistido por IA (opcional).** [`build_from_prompt`](@ref) traduce una descripción en
  lenguaje natural a un TOML ejecutable mediante un LLM local.

## Instalación

```julia
using Pkg
Pkg.add(url = "https://github.com/celhersot/CellIA.jl")
```

## Un vistazo rápido

```julia
using CellIA, TOML

config = TOML.parsefile("examples/organismo.toml")  # describe el modelo
model  = initialize_model(config)                    # construye y puebla el mundo
video_simulation(model, config["visualization"], config["space"])  # genera el MP4
```

O directamente desde la línea de comandos con el lanzador de ejemplos:

```bash
julia examples/main.jl examples/organismo.toml
```

## Contenido

```@contents
Pages = ["guia.md", "api.md"]
Depth = 2
```
