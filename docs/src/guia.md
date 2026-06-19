# Guía de uso

## Inicio rápido

La forma más directa de ejecutar un modelo es el lanzador `examples/main.jl`, que lee un TOML
y, opcionalmente, un archivo de reglas de usuario:

```bash
# Solo configuración
julia examples/main.jl examples/organismo.toml

# Configuración + reglas de usuario (modelos que aportan su propio struct/funciones)
julia examples/main.jl examples/lenia_perturbation.toml examples/lenia_perturbation.jl
```

Programáticamente el flujo es: leer el TOML → [`initialize_model`](@ref) → pedir una salida
([`video_simulation`](@ref), [`photo_simulation`](@ref) o [`run_simulation`](@ref)).

## Estructura de un TOML

Un modelo se describe con estas secciones (no todas son obligatorias):

| Sección | Para qué sirve |
|---|---|
| `[simulation]` | Nombre del modelo y `seed` (reproducibilidad). |
| `[space]` | `type` (`grid`/`continuous`/`hexagonal`), `dimensions`, `periodic`, `metric`. |
| `[agents]` | `state_type` (`Bool`, `Int`, `Symbol`, `Float64` o un tipo propio de las reglas). |
| `[population]` | `pop_density` o `pop_quantity` por estado: cuántos agentes y de qué tipo. |
| `[properties]` | Parámetros globales del modelo (p.ej. `lenia_mu`, `min_to_live`…). |
| `[rules]` | `initialization_rule`, `agent_step`, `model_step`, `post_init` (nombres de funciones en `CustomEvolutionRules`). |
| `[visualization]` | Salida de vídeo/foto: `filename`, `frames`, `framerate`, `color_scheme`, `title`. |
| `[run]` | Recogida de datos: `steps`, `output` (CSV) y `mdata`/`adata` (propiedades o funciones agregadoras). |

Ejemplo mínimo (Game of Life):

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

## Catálogo de modelos de ejemplo

En `examples/` hay configuraciones listas para ejecutar:

| Modelo | TOML | Tipo |
|---|---|---|
| Game of Life | `conway.toml`, `gol.toml` | grid booleano |
| Rock–Paper–Scissors | `rps.toml` | grid de símbolos |
| Segregación de Schelling | `schelling.toml` | grid con movimiento |
| Incendio forestal | `forestFire.toml` | grid |
| Flocking / boids | `flocking.toml` | espacio continuo |
| Lenia (orbium) | `lenia.toml`, `organismo.toml` | grid de campo continuo (FFT) |
| Lenia bajo perturbación | `lenia_perturbation.toml` | + ruido estocástico y métricas de energía |
| Image trace | `image_trace.toml`, `image_trace_pointillist.toml` | agentes que reproducen una imagen |
| Otros | `isla.toml`, `hive.toml`, `particles.toml` | varios |

## Cómo añadir un modelo nuevo

1. **Estado del agente:** usa un tipo básico en `[agents].state_type` o define tu propio struct
   en un archivo de reglas (como `Painter` en `examples/image_trace.jl`).
2. **Reglas:** escribe la(s) función(es) de actualización (`agent_step!` y/o `model_step!`) y, si
   necesitas preparación inicial, un `post_init`. Pueden ir en `CustomEvolutionRules` (genéricas)
   o en un archivo de usuario que `main.jl` inyecta.
3. **Configuración:** crea el `.toml` nombrando esas reglas en `[rules]`.
4. **Ejecuta:** `julia examples/main.jl tu_modelo.toml [tus_reglas.jl]`.
