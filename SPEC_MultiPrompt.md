# Spec: Sistema multi–system-prompt con router-agente local (Qwen2.5-Coder-14B)

## 0. Decisiones cerradas (acordadas con la usuaria)

1. **Backend**: modelo **local** vía `llama.cpp` (sin coste, offline tras la descarga inicial). Claude descartado (no existe en local/gratis). **Integración por `llama-server` (HTTP en `localhost`) — confirmado por la usuaria**; `llama-cli` queda solo como fallback de emergencia.
2. **Modelo**: **Qwen2.5-Coder-7B-Instruct** Q4_K_M (~4.36 GB). *(Se cambió de 14B a 7B al detectar el hardware real: Intel i7-8565U + GeForce MX130 de 2 GB VRAM; un 14B correría en CPU y sería muy lento.)* **Un único modelo** hace router y generación, **en CPU** (la GPU de 2 GB no aloja capas útiles).
3. **Router = agente LLM** que razona sobre la frase completa (abstracta, multilingüe). **NO** hay clasificador por palabras clave.
4. **4 categorías**: `grid_discrete`, `continuous_field` (Lenia), `continuous_space` (flocking/partículas), `hexagonal`.
5. **Modularizar el espacio `continuous`**: nada hardcodeado (hoy lo está). La forma "bird" (flecha orientada) pasa a ser un marcador reutilizable en `Representation`.
6. **Bucle validar → reparar** es **núcleo** (no opcional): es lo que más sube el "a la primera".
7. `predator_prey` se ignora.

> **Expectativa realista (honesta).** Ningún modelo local garantiza `.jl` "sin errores" siempre — y menos en **Julia, que está poco representada** en los datos de entrenamiento. Lo que hace fiable el sistema: (a) las categorías con reglas *built-in* no necesitan `.jl`; (b) prompts casi-plantilla por categoría; (c) bucle de reparación. Estimación: **~85–95 % de éxito a la primera en los casos canónicos** de las 4 categorías; peticiones exóticas pueden requerir un reintento o retoque.

---

## 1. Cambio de modelo (Qwen2.5-Coder-14B)

### 1.1 Constantes (`LLMBuilder.jl`)
```julia
const MODEL_NAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
const MODEL_URL  = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
```
> El 7B Q4_K_M es **un único `.gguf` de 4.36 GB** (verificado en HuggingFace; no partido). Descarga única con wifi; luego funciona **offline**.

### 1.2 Plantilla de chat
Qwen2.5-Coder usa **ChatML** (`<|im_start|>system … <|im_end|>` / `user` / `assistant`), que es **exactamente** lo que ya monta `run_llama` hoy. No cambia.

### 1.3 Integración: `llama-server` (recomendado) vs `llama-cli` (actual)
El flujo nuevo hace **varias llamadas** por petición (router + generación + 1–3 reparaciones). Recargar 9 GB por llamada con `llama-cli` es lento.

- **Recomendado — `llama-server`**: arranca el modelo **una vez** y queda residente sirviendo en `http://localhost:8080`. Julia llama por HTTP (`HTTP.jl` + `JSON.jl`) al endpoint `/completion` (o `/v1/chat/completions`). 100 % local y gratis; el agente va fluido. Requiere lanzar el server (script `bin/start_server` o arranque automático desde Julia con `run(detach(...))`).
- **Alternativa — `llama-cli`**: se mantiene la arquitectura actual (un proceso por llamada). Más simple de arrancar, pero recarga el modelo cada vez. Sirve para una demo si no se quiere gestionar un proceso server.

> **Decisión confirmada: `llama-server`.** Toda llamada pasa por una función `llm_call(system, user; grammar, n_predict)` que encapsula el HTTP a `localhost`. `llama-cli` queda solo como fallback de emergencia dentro de esa misma función. El resto del código no se entera del backend.

### 1.4 Flags de inferencia (CPU; MX130 de 2 GB)
- `--ctx-size 8192` (suficiente; cabe en RAM con el 7B).
- `--n-gpu-layers 0` + `--threads 4` (núcleos físicos del i7-8565U): con 2 GB de VRAM la GPU no aloja capas útiles del 7B, así que el cómputo va a CPU. (Si algún día hay GPU con más VRAM, subir `--n-gpu-layers`.)
- Router: `--n-predict 256` (salida JSON corta). Generación: `--n-predict 4096`.

---

## 2. Taxonomía de categorías (conjunto cerrado de 4)

Claves internas en `snake_case` (las que emite el router y mapean a un archivo de prompt).

### 2.1 `grid_discrete`
- **Cubre**: autómatas de estado discreto en rejilla cuadrada — Game of Life, RPS, Schelling, fuego forestal, epidemias en rejilla, islas.
- **Espacio**: `type = "grid"` (`GridSpaceSingle`). **`state_type`**: `Bool`|`Int`|`Symbol`|`String`|`Float64` (discreto por umbrales).
- **Reglas**: `agent_step!` + `model_step = "default_model_step!"`; actualización síncrona vía `model.next_states[agent.id] = nuevo`.
- **`.jl`**: opcional — con built-in (`gol_step!`, `rps_step!`, `schelling_step!`) **no hace falta**.
- **Convención `.jl`**: **NO** `using`/`import`/`@agent`; solo funciones `step!`.
- **Viz**: `color_scheme` Dict; `agent_shape` `"rect"`/`"circle"`/Dict.

### 2.2 `continuous_field` (Lenia / campo continuo)
- **Cubre**: autómatas de **valor continuo** `Float64 ∈ [0,1]` por celda con convolución de kernel grande — Lenia, SmoothLife, reacción-difusión.
- **Espacio**: `type = "grid"`, `periodic = true`. **`state_type`**: `Float64`.
- **Reglas**: **solo** `model_step = "lenia_model_step!"`; `post_init = "lenia_init!"` (precomputa kernel FFT); `initialization_rule = "uniform_float"`.
- **`.jl`**: opcional (built-ins bastan para Lenia estándar; un `.jl` solo para redefinir `kernel_fn`/`growth_fn`).
- **Viz**: `color_scheme = "viridis"` → heatmap (`Representation.record_grid_heatmap`).
- **Propiedades**: `lenia_mu`, `lenia_sigma`, `dt`, `kernel_radius`, `kernel_type`.

### 2.3 `continuous_space` (flocking / partículas)
- **Cubre**: agentes que se **mueven** en espacio continuo 2D — boids/flocking, partículas, enjambres, presa-depredador con movimiento.
- **Espacio**: `type = "continuous"`. **`state_type`**: struct propio que subtipa `ContinuousAgent{2,Float64}`.
- **Reglas**: `agent_step!` con `nearby_agents(a, model, radio)`, vectores `SVector`, `move_agent!(a, model, speed)`.
- **`.jl`**: **OBLIGATORIO**: `using Agents`, `using Random, LinearAlgebra`, el `@agent struct ...`, y `agent_step!`.
- **Viz**: `agent_shape = "arrow"` (flecha orientada por velocidad — ver §9.3); color por tipo o `state.campo`.
- **✔ Ya NO está acoplado a `Bird`** (gracias a §9): el struct puede tener **los campos que quiera**; basta con que `[agents]` del TOML dé un valor para cada campo. El prompt debe instruir: definir el `@agent struct` con sus `const` campos, y listar **esos mismos campos con valor** bajo `[agents]`. `visual_distance` pasa a ser una propiedad normal que lee la regla, no una exigencia del framework.

### 2.4 `hexagonal`
- **Cubre**: modelos en rejilla hexagonal con propiedades por celda — colmena, propagación en panal.
- **Espacio**: `type = "hexagonal"` (`HexagonalGridSpace`). **`state_type`**: `Bool`|`Int`|`Symbol`.
- **Reglas**: `agent_step!` con `nearby_positions(agent.pos, model)`, `move_agent!`, y propiedades de celda vía `abmspace(model).cell_properties[agent.pos]`.
- **`.jl`**: necesario (no hay built-in hexagonal). **Convención**: como `grid_discrete` (sin `@agent`).
- **Viz**: `agent_shape = "hexagon"`, `cell_color_property` (+ `cell_color_max`), `color_scheme` Dict.

> El conflicto que justifica separar prompts: `grid_discrete`/`hexagonal` **prohíben** `@agent`/`using`, mientras `continuous_space` los **exige**. Meterlos juntos es lo que confunde al modelo.

---

## 3. Arquitectura del flujo

```
usuario: frase (cualquier idioma, abstracta)
        │
        ▼
 [1] ROUTER-AGENTE  (LLM, salida JSON con gramática GBNF)
        │  → {category, approach, reason}
        ▼
 [2] CONFIRMACIÓN (CLI): muestra `approach` legible
        │  [s] confirma → [3]
        │  [n] / texto correctivo → re-router excluyendo categoría rechazada → vuelve a [2]
        ▼ (sí)
 [3] GENERACIÓN: system = common_core + prompt(category) → genera .toml (+ .jl)
        ▼
 [4] VALIDACIÓN ESTÁTICA (sin ejecutar): esquema TOML + reglas por categoría
        │  ── falla ──► reinyecta error → regenera (máx 2)
        ▼ (ok)
 [5] EJECUCIÓN: julia main.jl <toml> [<jl>]
        │  ── crashea ──► captura stderr de Julia → reinyecta → regenera (máx 1–2)  [REPARACIÓN POR EJECUCIÓN]
        ▼ (ok)
   vídeo en output_videos/
```

Todo es **una sola fase** de entrega (router + confirm + generación + validación/reparación + ejecución).

---

## 4. Archivos de prompts

```
src/prompts/
  common_core.txt        # formato de salida, naming, esqueleto TOML, reglas duras universales
  router.txt             # prompt del router-agente (instrucciones de clasificación + contrato JSON)
  router.gbnf            # gramática que fuerza el JSON del router (categoría ∈ las 4)
  grid_discrete.txt
  continuous_field.txt
  continuous_space.txt
  hexagonal.txt
```
**Composición**: `system_prompt(cat) = read(common_core.txt) * "\n\n" * read("$(cat).txt")`.
`common_core.txt` = las ~30 primeras líneas del `system_prompt.txt` actual (formato `FILENAME:` + bloques ```` ``` ````, naming, esqueleto de secciones, reglas duras `abmspace`/`abmrng`/`rand(abmrng(model))`).

---

## 5. Router-agente

### 5.1 Contrato de salida (forzado por gramática)
El router devuelve **solo** este JSON:
```json
{ "category": "grid_discrete | continuous_field | continuous_space | hexagonal",
  "approach":  "frase legible para el usuario: qué se va a montar",
  "reason":    "por qué esta categoría" }
```

### 5.2 Fiabilidad con gramática GBNF (clave en local)
`llama.cpp` permite `--grammar-file router.gbnf` para **constreñir** la salida: garantiza JSON válido y que `category` sea **exactamente una de las 4**. Es el equivalente local a los *structured outputs*. Esquema de `router.gbnf`:
```gbnf
root     ::= "{" ws "\"category\"" ws ":" ws category ws "," ws
                  "\"approach\"" ws ":" ws string ws "," ws
                  "\"reason\""   ws ":" ws string ws "}"
category ::= "\"grid_discrete\"" | "\"continuous_field\"" | "\"continuous_space\"" | "\"hexagonal\""
string   ::= "\"" ( [^"\\] | "\\" . )* "\""
ws       ::= [ \t\n]*
```
> Si se usa `llama-server`, el endpoint nativo acepta el campo `grammar` en el JSON de la petición — mismo efecto sin fichero.

### 5.3 Re-routing tras rechazo
Al rechazar, se reinvoca el router con:
- la lista de categorías **excluyendo** la(s) ya rechazada(s) (se inyecta en el prompt: "NO uses estas categorías: …"),
- el **texto correctivo** del usuario concatenado a la descripción original.

Si solo queda una categoría, se propone directamente sin volver a llamar al modelo.

### 5.4 Parsing robusto
Aun con gramática, extraer el **primer bloque `{...}`** y `JSON.parse`. Si fallara (no debería con gramática), reintentar una vez; si persiste, abortar con mensaje claro (sin fallback de keywords, por decisión de diseño).

---

## 6. Confirmación (CLI)

`build_from_prompt` es interactivo (`readline(stdin)`):
```
--> Propuesta: <approach>
    (categoría interna: <category>)
¿Generar con este enfoque? [s / n / o escribe una corrección]:
```
- vacío / `s` / `si` / `sí` / `y` → confirmar → [3].
- `n` / `no` → re-router excluyendo la categoría actual → nueva propuesta.
- cualquier otro texto → corrección: re-router con ese texto añadido.
- Límite de 3 rechazos → aborta con mensaje.
- Flag `--yes` (no interactivo, p.ej. pruebas): salta confirmación y usa la categoría propuesta.

---

## 7. Generación

`llm_call(system_prompt, user_text; n_predict=4096)` con `system_prompt = common_core + prompt(category)`.
Formato de salida **se mantiene** (`FILENAME:` + bloques de código) y se reusa `parse_llm_response`. (No se fuerza el código con gramática: serializar Julia multilínea dentro de JSON es propenso a errores de escape; el formato con bloques es más fiable para un modelo de código.)

---

## 8. Validación + reparación (núcleo)

### 8.1 Validación estática (sin ejecutar Julia)
**TOML (todas)**: parsea con `TOML.parse`; tiene `[simulation] [space] [agents] [rules] [visualization]`; `space.type` == el esperado por la categoría; `state_type` válido; `rules.*` referencian funciones existentes (built-in) o definidas en el `.jl` generado (regex `function\s+nombre!`).

**Por categoría**:
- `continuous_field`: `state_type == "Float64"`, `model_step == "lenia_model_step!"`, existe `post_init`, `color_scheme` es string de paleta.
- `continuous_space`: existe `.jl`; el `.jl` contiene `@agent` y `using Agents`; **cada campo del struct tiene valor en `[agents]`**; `population.pop_quantity` presente.
- `grid_discrete`: si `agent_step` no es built-in → definido en el `.jl`; el `.jl` **no** contiene `using`/`@agent`.
- `hexagonal`: `space.type == "hexagonal"`, existe `.jl`, `agent_shape == "hexagon"`.

### 8.2 Bucle de reparación estática
```
intentos = 0
while !(ok, _ = validate(files, cat))[1] && intentos < 2
    files = llm_call(prompt(cat) * "\nEl intento anterior falló: $msg_error. Corrige SOLO eso y vuelve a emitir los archivos.", user_text)
    intentos += 1
end
```

### 8.3 Reparación por error de ejecución (lo más potente)
Tras pasar validación estática, se ejecuta `julia main.jl <toml> [<jl>]` **capturando stderr**. Si Julia crashea, su mensaje de error es **muy preciso** (línea, función, tipo). Se reinyecta:
```
"Al ejecutar, Julia dio este error:\n<stderr>\nCorrige los archivos para que ejecute sin error."
```
y se regenera (máx 1–2 intentos). Si sigue fallando: se guardan los archivos + `llm_raw_output.txt` y se avisa con el comando para ejecutar a mano.

---

## 9. Modularización del espacio continuo (punto 3)

Objetivo: **nada hardcodeado**; cualquier autómata continuo (no solo flocking) funciona.

### 9.1 `Initialization.populate_continuous_world!` — quitar campos fijos
Hoy ([Initialization.jl:114-143](src/Initialization.jl#L114-L143)) instancia con `agents_conf["speed"], agents_conf["cohere_factor"], …` (fijo a `Bird`). Cambiar a **reflexión**:
```julia
# Campos propios del struct = fieldnames(T) menos los que @agent añade (id, pos, vel)
extra = filter(f -> f ∉ (:id, :pos, :vel), collect(fieldnames(T)))
for _ in 1:qty
    vel    = rand(abmrng(model), SVector{2,Float64}) .* 2 .- 1
    kwargs = Dict(f => convert_type(agents_conf[string(f)], fieldtype(T, f)) for f in extra)
    add_agent!(model; vel = vel, kwargs...)
end
```
Así el struct puede tener cualquier conjunto de campos; el TOML solo debe proveer un valor por campo en `[agents]`.

### 9.2 `SpaceDefinition.create_space` continuo — `visual_distance` opcional
Hoy ([SpaceDefinition.jl:18](src/SpaceDefinition.jl#L18)) **exige** `agents.visual_distance`. Cambiar a opcional con default de espaciado:
```julia
spacing = haskey(space_conf, "spacing") ? space_conf["spacing"] :
          (haskey(config["agents"], "visual_distance") ? config["agents"]["visual_distance"]/1.5 : 1.0)
return ContinuousSpace(extent; periodic = periodic, spacing = spacing)
```

### 9.3 `Representation` — la forma "bird" como marcador reutilizable
Hoy `bird_polygon`/`marker_shape` existen y `"marker"` está en `SPECIAL_FUNCTIONS` ([Representation.jl:9-24](src/Representation.jl#L9-L24)). Acción: **renombrar/registrar como `"arrow"`** (flecha orientada por `vel`) y dejarlo seleccionable desde cualquier TOML continuo (`agent_shape = "arrow"`). Es visualización reutilizable, no inicialización — exactamente lo que pediste. Mantener `"marker"` como alias para no romper `flocking.toml`.

> Estos cambios **relajan** lo que el prompt `continuous_space` debe exigir: ya no hay que ceñirse a los campos de `Bird`.

---

## 10. Cambios por archivo

| Archivo | Acción | Detalle |
|---|---|---|
| `src/prompts/common_core.txt` | **Crear** | Cabecera compartida (de `system_prompt.txt`). |
| `src/prompts/router.txt` + `router.gbnf` | **Crear** | Router-agente + gramática JSON (§5). |
| `src/prompts/{grid_discrete,continuous_field,continuous_space,hexagonal}.txt` | **Crear** | Cuerpo + 1 ejemplo cada uno (reusar `gol`/`lenia`/`flocking`/`hive`). |
| `src/LLMBuilder.jl` | **Modificar** | Modelo 14B; `llm_call` (HTTP a llama-server / fallback cli); `CATEGORIES`, `load_prompt`, `route`, `confirm_loop`, `validate_files`, reparación; `build_from_prompt` interactivo (§11). |
| `src/Initialization.jl` | **Modificar** | `populate_continuous_world!` por reflexión (§9.1). |
| `src/SpaceDefinition.jl` | **Modificar** | `visual_distance` opcional (§9.2). |
| `src/Representation.jl` | **Modificar** | Marcador `"arrow"` reutilizable (§9.3). |
| `Project.toml` | **Modificar** | Añadir `HTTP` y `JSON` (para `llama-server`). |
| `bin/start_server.*` | **Crear (opc.)** | Lanzar `llama-server` con el modelo y flags de §1.4. |
| `examples/llm.jl` | **Modificar** | Flujo interactivo + flag `--yes`. |
| `src/system_prompt.txt` | Conservar | Legacy hasta validar el nuevo flujo. |

---

## 11. Funciones nuevas en `LLMBuilder.jl`

```julia
const CATEGORIES = Dict(
  "grid_discrete"    => (prompt="grid_discrete.txt",    space="grid",       needs_jl=false),
  "continuous_field" => (prompt="continuous_field.txt", space="grid",       needs_jl=false),
  "continuous_space" => (prompt="continuous_space.txt", space="continuous", needs_jl=true),
  "hexagonal"        => (prompt="hexagonal.txt",         space="hexagonal",  needs_jl=true),
)
const PROMPTS_DIR = joinpath(@__DIR__, "prompts")

llm_call(system, user; grammar=nothing, n_predict=4096)  # HTTP→llama-server, fallback llama-cli
load_prompt(category)                                    # common_core + prompts/<category>.txt
route(user_text; exclude=String[])                       # → (category, approach, reason)  [usa router.gbnf]
confirm_loop(route_result)                               # CLI interactiva → categoría confirmada | :abort
validate_files(files, category)                          # → (ok::Bool, error_msg::String)
build_from_prompt(user_text; interactive=true, output_dir=...)  # orquesta [1]..[5] + reparación
```

---

## 12. Realismo y límites (a tener presente)

- **Julia es de bajo recurso** en los modelos → el `.jl` es el punto débil; por eso se prioriza built-ins, prompts-plantilla y reparación por error de ejecución (§8.3).
- **Hardware real (i7-8565U, 16 GB RAM, MX130 2 GB)**: se usa el 7B en CPU. Estimación: routing <1 min, generación ~2–5 min. Lento pero usable.
- **Primera ejecución**: descarga de ~4.36 GB (una vez, con wifi). Después, **offline**.
- **`predator_prey`** fuera de scope.

---

## 13. Plan de pruebas

**Router (frase → categoría):**
| Frase | Esperado |
|---|---|
| "el juego de la vida de Conway" | `grid_discrete` |
| "segregación de dos grupos que se mudan si están rodeados" | `grid_discrete` |
| "bichos continuos tipo Lenia que se mueven solos" | `continuous_field` |
| "reacción-difusión con un kernel de convolución" | `continuous_field` |
| "una bandada de pájaros que vuelan juntos" | `continuous_space` |
| "partículas que se agrupan moviéndose en 2D" | `continuous_space` |
| "una colmena de abejas en un panal hexagonal" | `hexagonal` |
| "propagación en una rejilla hexagonal" | `hexagonal` |

**Generación end-to-end** (debe producir vídeo sin error): 1 caso canónico por categoría.
**Reparación**: inyectar un error a propósito en un `.toml`/`.jl` y comprobar que se repara en ≤2 intentos (estática) y que un error de ejecución se corrige (§8.3).
**Modularización continua**: definir un `@agent` con campos **distintos** a `Bird` y confirmar que se puebla y ejecuta.
**Rechazo**: rechazar una propuesta y comprobar que cambia de categoría.

**Criterios de aceptación**: las 8 frases enrutan bien; los 4 casos canónicos ejecutan; el rechazo reenruta; la reparación funciona; un continuo no-`Bird` corre.

---

## 14. Orden de implementación sugerido

1. **Modelo + `llm_call`**: cambiar a 14B; montar `llama-server` + `llm_call` HTTP (fallback `llama-cli`). Verificar una llamada simple.
2. **Modularizar `continuous`** (§9): `Initialization`, `SpaceDefinition`, `Representation`. Probar `flocking` y un struct no-`Bird`.
3. **Prompts**: `common_core` + 4 categorías (reusando ejemplos) + `router.txt`/`router.gbnf`.
4. **Router** (`route` + gramática) → pasar el test de las 8 frases con `--yes`.
5. **Confirmación** (`confirm_loop`) + `build_from_prompt` interactivo.
6. **Validación + reparación** (§8) — estática y por ejecución.
7. Pasar todos los criterios de §13.

---

## 15. Fuera de scope

- Generalizar más allá de 2D en continuo.
- Multi-canal en Lenia / Lenia hexagonal.
- Cualquier backend de pago o que requiera internet en ejecución.
- `predator_prey`.
- Routing por embeddings (el router-agente LLM es suficiente para 4 clases).
```
