# Spec: Lenia en el framework Cell_IA

## Qué es Lenia y por qué es diferente

Lenia es un autómata celular **continuo**: cada celda tiene un estado en `[0, 1]` (Float64) y evoluciona según:

```
A(t+Δt) = clip( A(t) + Δt · G( K ⊛ A(t) ), 0, 1 )
```

donde `⊛` es convolución, `K` es un kernel suave (función radial), y `G` es una función de crecimiento (campana gaussiana). Esto lo diferencia de GoL o RPS en dos puntos clave:

| Aspecto | GoL / RPS | Lenia |
|---|---|---|
| Estado por celda | discreto (`Bool`, `Symbol`) | continuo (`Float64` ∈ [0,1]) |
| Actualización | local (vecinos inmediatos) | convolución global con radio R grande (típico R=13) |
| Rendimiento del naive | aceptable | catastrófico con `nearby_agents` (O(N·R²)) |

---

## Lo que ya funciona sin tocar

- **Espacio**: `type = "grid"`, `periodic = true` → `GridSpaceSingle`. Lenia vive en una malla discreta, no necesita espacio nuevo.
- **Agente**: `UniversalAgent{Float64}` con `state::Float64` ya existe.
- **Actualización síncrona**: el mecanismo `next_states` + `default_model_step!` es compatible.
- **FFTW**: ya está en `Project.toml`. La convolución eficiente está disponible.
- **Configuración TOML**: los parámetros `mu`, `sigma`, `dt`, `kernel_radius` van en `[properties]` como cualquier otro modelo.

---

## Problemas del `lenia_step!` actual

El `lenia_step!` en `CustomEvolutionRules.jl` tiene tres fallos:

1. Usa `parse(Float64, string(agent.state))` — innecesario si el tipo ya es `Float64`.
2. Usa `nearby_agents` con radio implícito 1 — no es convolución real con radio grande.
3. Referencia `agent.future_state` que no existe en `UniversalAgent`.

Se puede reemplazar con la arquitectura correcta sin tocar la firma del TOML.

---

## Arquitectura propuesta

### Núcleo de la implementación

Toda la lógica va en **`model_step!`**, no en `agent_step!` (que queda como `dummystep`).

```
lenia_model_step!(model):
    1. Extraer estados → matriz A[rows, cols]  (O(N))
    2. Convolucionar:  U = real(ifft(fft(A) .* model.kernel_fft))  (O(N log N))
    3. Aplicar growth: G(u) = 2·exp(-((u-μ)²)/(2σ²)) - 1
    4. Actualizar:     model.next_states[id] = clip(A + Δt·G, 0, 1)
    5. default_model_step! aplica next_states → agents
```

El kernel FFT (`model.kernel_fft`) se **precomputa una sola vez** y se guarda en `model.properties`.

### Extensión necesaria en `Initialization.jl`: hook `post_init`

Un hook genérico que, si el TOML define `rules.post_init`, llama a esa función sobre el modelo **después** de crear y poblar el mundo:

```julia
# en initialize_model, al final de cada branch de space_type:
if haskey(rules_conf, "post_init")
    post_init_fn = getfield(CustomEvolutionRules, Symbol(rules_conf["post_init"]))
    Base.invokelatest(post_init_fn, model)
end
```

Esto es genérico: sirve para Lenia (precomputar kernel), para la colmena (inicializar miel), o para cualquier otro modelo que necesite preparación post-creación.

### Nueva `initialization_rule`: `"uniform_float"`

Para Lenia, **todas las celdas** se inicializan con un float aleatorio en `[0, 1]`. Se añade como nuevo caso en `populate_world!`:

```julia
elseif init_rule == "uniform_float"
    for pos in positions(model)   # itera todas las posiciones
        val = rand(abmrng(model))
        add_agent!(pos, model; state = val)
    end
```

Esto es genérico y útil para cualquier autómata continuo.

---

## Interfaz modular del kernel

El usuario define en `lenia.jl` dos funciones opcionales:

```julia
# Función de kernel radial (recibe r normalizado en [0,1])
kernel_fn(r) = max(0.0, 4r*(1-r))^4  # polinómica tipo Lenia por defecto

# Función de crecimiento (recibe u = (K ⊛ A)(x))
growth_fn(u, μ, σ) = 2.0*exp(-((u - μ)^2)/(2σ^2)) - 1.0  # gaussiana
```

La función `lenia_init!` del rules file:
1. Lee `kernel_radius`, `kernel_type` del TOML (vía `model.properties`)
2. Llama a `kernel_fn` o usa la predeterminada según `kernel_type`
3. Normaliza y hace la FFT del kernel
4. Guarda en `model.kernel_fft`

El usuario puede redefinir `kernel_fn` y `growth_fn` para explorar nuevas criaturas.

---

## TOML de ejemplo (`lenia.toml`)

```toml
[simulation]
model_name = "Lenia"
seed = 42

[space]
type = "grid"
dimensions = [128, 128]
periodic = true
metric = "chebyshev"

[agents]
state_type = "Float64"

[population]
pop_density = { "0.0" = 1.0 }     # poblar todas las celdas; post_init randomiza

[properties]
lenia_mu     = 0.15
lenia_sigma  = 0.015
dt           = 0.1
kernel_radius = 13
kernel_type  = "gaussian"          # "gaussian" | "polynomial" | "custom"

[rules]
post_init    = "lenia_init!"       # precomputa kernel + randomiza estados
model_step   = "lenia_model_step!"
initialization_rule = "random"

[visualization]
filename     = "output_videos/lenia.mp4"
frames       = 200
framerate    = 20
title        = "Lenia"
variable_to_color = "state"
color_scheme = "viridis"           # nuevo: soporte de paletas continuas en Representation
agent_shape  = "rect"
agent_size   = 1
```

---

## Archivos a crear / modificar

| Archivo | Acción | Qué cambia |
|---|---|---|
| `src/Initialization.jl` | Modificar | Añadir hook `post_init` + `initialization_rule = "uniform_float"` |
| `src/CustomEvolutionRules.jl` | Modificar | Reemplazar `lenia_step!` roto por `lenia_model_step!` + `lenia_init!` genéricos |
| `src/Representation.jl` | Modificar | Soporte de `color_scheme = "viridis"` (paleta continua para Float64 states) |
| `examples/lenia.toml` | Crear | Config estándar de Lenia |
| `examples/lenia.jl` | Crear | `lenia_init!`, `lenia_model_step!`, kernel y growth modulares |

Sin archivos nuevos en `src/` — Lenia encaja completamente en la infraestructura existente de `grid`.

---

## Decisiones de diseño y alternativas

### FFT vs. `nearby_agents`

| | FFT (`fft` + `ifft`) | `nearby_agents` |
|---|---|---|
| Complejidad | O(N log N) | O(N · R²) |
| Grid 128×128, R=13 | ~1 ms | ~170 ms |
| Grid 512×512, R=13 | ~5 ms | ~2.7 s |
| Requiere | FFTW (ya en deps) | nada |

FFT es obligatorio para cualquier tamaño útil.

### Kernel storage

El kernel FFT se guarda en `model.properties[:kernel_fft]` (tipo `Matrix{ComplexF64}`). Se computa una vez en `post_init`, no en cada step.

### Gradient descent hacia `next_states`

El mecanismo `next_states` + `default_model_step!` añade O(N) iteraciones extra. Para grids grandes, es posible hacer un update directo en `lenia_model_step!` sin `next_states`. **Se puede implementar como opción** añadiendo `model_step = "lenia_model_step_fast!"` que hace el update directo. Se especifica en el spec pero se decide durante la implementación.

### Paleta continua en `Representation.jl`

Actualmente `color_scheme` es un `Dict` o `Vector` para estados discretos. Para `Float64` en `[0, 1]`, se añade soporte para nombres de paletas Makie (`"viridis"`, `"inferno"`, etc.) que mapean linealmente el estado al color.

---

## Fuera del scope de esta tarea

- Multi-channel Lenia (varios canales de estado por celda)
- Lenia en espacio hexagonal
- Optimización con `PackageCompiler` para TTFP
