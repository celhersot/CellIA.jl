module LLMBuilder

# Local-LLM driven generator for Cell_IA simulations.
#
# Flujo:
#   1. route()                -> router-agent classifies the request into one of 4 categories
#   2. confirm (CLI)          -> user accepts / rejects / corrects
#   3. generate + validate    -> per-category system prompt, static validation, repair loop
#   4. run + repair-on-error  -> execute; on Julia error, feed stderr back and regenerate
#
# Backend: llama-cli (bin/llama-cli.exe), one-shot via -f prompt file, no server.

using Downloads
using TOML
using JSON

export build_from_prompt, route, validate_files, load_prompt

# --- Model configuration ---

const MODEL_DIR  = joinpath(@__DIR__, "..", "models")
const MODEL_NAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
const MODEL_PATH = joinpath(MODEL_DIR, MODEL_NAME)
# 1.5B (~1 GB): small enough to run on CPU in a couple of minutes on any laptop.
const MODEL_URL  = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

const PROMPTS_DIR = joinpath(@__DIR__, "prompts")

# category => (prompt file, expected space.type, whether a _rules.jl is mandatory,
#              canonical human description shown at confirmation time)
# `desc` is a curated, coherent sentence per category. We show THIS instead of the
# router's free-form "approach": the 1.5B model often writes contradictory approaches
# (e.g. "Lenia con cada célula viva o muerta"), which makes a correct category look wrong.
const CATEGORIES = Dict(
    "grid_discrete"    => (prompt = "grid_discrete.txt",    space = "grid",       needs_jl = false,
        desc = "Autómata celular en rejilla: cada celda tiene un estado DISCRETO (p.ej. viva/muerta, un color, una especie) y se actualiza según sus vecinas. Las celdas no se mueven. Estilo Juego de la Vida."),
    "continuous_field" => (prompt = "continuous_field.txt", space = "grid",       needs_jl = false,
        desc = "Tipo Lenia: una rejilla cuyas celdas guardan un valor CONTINUO en [0,1]; el conjunto forma un organismo que fluye y se desplaza de forma suave sobre la rejilla."),
    "continuous_space" => (prompt = "continuous_space.txt", space = "continuous", needs_jl = true,
        desc = "Tipo bandada/boids: MUCHOS agentes se mueven como puntos por un espacio continuo (sin rejilla) y se agrupan según sus vecinos."),
    "hexagonal"        => (prompt = "hexagonal.txt",         space = "hexagonal",  needs_jl = true,
        desc = "Modelo sobre rejilla HEXAGONAL (panal), con propiedades por celda. Estilo colmena."),
)

const BUILTIN_RULES = Set([
    "gol_step!", "rps_step!", "schelling_step!", "default_model_step!",
    "lenia_model_step!", "lenia_noisy_step!", "lenia_init!",
])

# Métricas de NIVEL MODELO (f(model)->escalar). En [run] van SIEMPRE en mdata, nunca en adata
# (en adata Agents.jl las llamaría por-agente y fallaría).
const MODEL_METRICS = Set(["total_energy", "mean_state", "active_cells"])

# Read once at load time (the files ship with the framework).
const ROUTER_GRAMMAR = read(joinpath(PROMPTS_DIR, "router.gbnf"), String)

# --- Prompt loading ---

function load_prompt(category::String)
    haskey(CATEGORIES, category) || error("Categoría desconocida: $category")
    core = read(joinpath(PROMPTS_DIR, "common_core.txt"), String)
    body = read(joinpath(PROMPTS_DIR, CATEGORIES[category].prompt), String)
    return core * "\n\n" * body
end

function load_router_prompt(exclude::Vector{String} = String[])
    base = read(joinpath(PROMPTS_DIR, "router.txt"), String)
    if !isempty(exclude)
        base *= "\n\nDO NOT use these categories (already rejected): " *
                join(exclude, ", ") * ". Pick a different one."
    end
    return base
end

# --- Model download (only needed for the llama-cli fallback) ---

function ensure_model_exists()
    isdir(MODEL_DIR) || mkpath(MODEL_DIR)
    if !isfile(MODEL_PATH)
        println("Descargando modelo (una sola vez, ~1 GB)...")
        try
            Downloads.download(MODEL_URL, MODEL_PATH)
            println("Modelo descargado.")
        catch e
            error("No se pudo descargar el modelo. Revisa la conexión: $e")
        end
    end
    return MODEL_PATH
end

# --- LLM call ---
# Runs llama-cli once (no server). The model is loaded, generates, and we capture the
# assistant turn from its output. Returns the assistant text only.
#
# Hard-won notes about THIS build (llama.cpp b9547) on Windows:
#   * --log-disable, -no-cnv and --simple-io all crash it (0xC0000409). Don't use them.
#   * On a small GPU (≤2 GB VRAM) full offload crashes intermittently while processing
#     the prompt. CPU is rock-solid, so we default to CPU. Users with a bigger GPU can
#     set CELL_IA_GPU_LAYERS=99 (or any number) to offload.
#   * In conversation mode it echoes the whole prompt and then waits for the next turn on
#     stdin. We feed it "/exit" so it quits cleanly (exit 0) after one generation. We then
#     extract only the text after the last <|im_start|>assistant marker (the prompt echo
#     contains our few-shot examples; parsing the raw output would pick those up instead).

const GPU_LAYERS = parse(Int, get(ENV, "CELL_IA_GPU_LAYERS", "0"))

function llm_call(system::String, user::String;
                  grammar::Union{Nothing,String} = nothing,
                  n_predict::Int = 4096)
    prompt = "<|im_start|>system\n$system<|im_end|>\n" *
             "<|im_start|>user\n$user<|im_end|>\n" *
             "<|im_start|>assistant\n"
    return _cli_completion(prompt; grammar = grammar, n_predict = n_predict)
end

# Pull out just the assistant's generated text: everything after the LAST assistant
# marker, trimmed at the trailing stats line. Falls back to the whole text.
function _extract_assistant(raw::AbstractString)
    clean = replace(String(raw), r"\e\[[0-9;]*m" => "")   # strip ANSI colours
    clean = replace(clean, "\r\n" => "\n")
    marker = "<|im_start|>assistant"
    idx = findlast(marker, clean)
    idx !== nothing && (clean = clean[nextind(clean, last(idx)):end])
    cut = findfirst("[ Prompt:", clean)                   # llama's end-of-turn stats
    cut !== nothing && (clean = clean[1:prevind(clean, first(cut))])
    return strip(clean)
end

function _cli_completion(prompt::String; grammar, n_predict, timeout_s::Int = 300)
    model_path   = ensure_model_exists()
    exe_name     = Sys.iswindows() ? "llama-cli.exe" : "llama-cli"
    llama_exe    = abspath(joinpath(@__DIR__, "..", "bin", exe_name))
    prompt_file  = abspath(joinpath(@__DIR__, "llm_input.txt"))
    grammar_file = abspath(joinpath(@__DIR__, "grammar.gbnf"))
    out_file     = abspath(joinpath(@__DIR__, "llm_output.txt"))
    stdin_file   = abspath(joinpath(@__DIR__, "llm_stdin.txt"))
    try
        write(prompt_file, prompt)
        # Conversation mode reads the next user turn from stdin after generating; feeding
        # "/exit" makes llama quit cleanly (exit 0) on its own; no hang, nothing to kill.
        write(stdin_file, "/exit\n")
        isfile(out_file) && rm(out_file; force = true)
        cmd = `$llama_exe --model $(abspath(model_path)) -f $prompt_file --n-gpu-layers $GPU_LAYERS --ctx-size 8192 --n-predict $n_predict --temp 0.2 --no-display-prompt`
        if grammar !== nothing
            write(grammar_file, grammar)
            cmd = `$cmd --grammar-file $grammar_file`
        end
        proc = run(pipeline(ignorestatus(cmd); stdin = stdin_file, stdout = out_file, stderr = devnull); wait = false)
        t0 = time()
        while process_running(proc) && (time() - t0 < timeout_s); sleep(1); end
        process_running(proc) && kill(proc)   # safety net if /exit ever fails to quit
        try; wait(proc); catch; end
        sleep(0.3)                            # let the output file handle flush/close
        raw = isfile(out_file) ? read(out_file, String) : ""
        return _extract_assistant(raw)
    catch e
        @error "Ejecución del modelo falló: $e"
        return ""
    finally
        for f in (prompt_file, grammar_file, out_file, stdin_file)
            try; isfile(f) && rm(f; force = true); catch; end
        end
    end
end

# --- Router ---

function parse_router_json(text::AbstractString)
    m = match(r"\{.*\}"s, text)
    m === nothing && return nothing
    local data
    try
        data = JSON.parse(m.match)
    catch
        return nothing
    end
    (data isa AbstractDict && haskey(data, "category")) || return nothing
    cat = String(data["category"])
    haskey(CATEGORIES, cat) || return nothing
    return (category = cat,
            approach = String(get(data, "approach", "")),
            reason   = String(get(data, "reason", "")))
end

function route(desc::String; exclude::Vector{String} = String[])
    system = load_router_prompt(exclude)
    for _ in 1:2
        raw = llm_call(system, desc; grammar = ROUTER_GRAMMAR, n_predict = 256)
        r = parse_router_json(raw)
        r !== nothing && return r
    end
    error("El router no devolvió un JSON válido tras 2 intentos.")
end

# --- Confirmation loop (CLI) ---
# Returns (category, final_description) or nothing if cancelled.

function confirm_and_select(user_text::String; interactive::Bool = true)
    excluded = String[]
    clarifications = String[]
    # Build the routing input from the ORIGINAL request plus any user clarifications,
    # kept as a labeled list rather than concatenated into one run-on sentence. Appending
    # corrections to a single string turns successive refinements ("rejilla", "movimiento
    # continuo") into a self-contradictory blob that confuses the router.
    build_desc() = isempty(clarifications) ? user_text :
        user_text * "\n\nAclaraciones del usuario (posteriores y más precisas; mandan sobre la " *
        "descripción inicial):\n" * join(["- " * c for c in clarifications], "\n")
    for _ in 1:6
        desc = build_desc()
        r = route(desc; exclude = excluded)
        if !interactive
            println("Categoría elegida automáticamente: $(r.category)")
            return (r.category, desc)
        end
        # Show the curated category description, not the router's noisy free-form approach.
        println("\nPropuesta: $(CATEGORIES[r.category].desc)")
        println("    (categoría interna: $(r.category))")
        print("¿Generar con este enfoque? [s / n / o escribe una corrección]: ")
        ans = strip(readline())
        low = lowercase(ans)
        if isempty(low) || low in ("s", "si", "sí", "y", "yes")
            return (r.category, desc)
        elseif low in ("n", "no")
            push!(excluded, r.category)
            if length(excluded) >= length(CATEGORIES)
                println("Se han descartado todas las categorías. Cancelando.")
                return nothing
            end
        else
            # Treat as a clarification and re-route WITHOUT excluding the current category:
            # a correction often just refines details (the right category may stay the same).
            # Only an explicit "n" rules a category out.
            push!(clarifications, String(ans))
        end
    end
    println("Demasiados intentos sin confirmar. Cancelando.")
    return nothing
end

# --- Static validation ---
# Returns (ok::Bool, error_msg::String). Does NOT execute Julia.

function validate_files(files::Dict, category::String)
    toml_str = ""
    jl_str   = ""
    for (fn, content) in files
        endswith(fn, ".toml") && (toml_str = content)
        endswith(fn, ".jl")   && (jl_str   = content)
    end

    isempty(toml_str) && return (false, "No se generó ningún archivo .toml.")

    local cfg
    try
        cfg = TOML.parse(toml_str)
    catch e
        return (false, "El .toml no es TOML válido: $e")
    end

    for sec in ("simulation", "space", "agents", "rules", "visualization")
        haskey(cfg, sec) || return (false, "Falta la sección [$sec] en el .toml.")
    end

    expected = CATEGORIES[category].space
    got      = get(get(cfg, "space", Dict()), "type", "")
    got == expected || return (false, "space.type debe ser \"$expected\" para $category (es \"$got\").")

    if CATEGORIES[category].needs_jl && isempty(jl_str)
        return (false, "La categoría $category requiere un archivo _rules.jl y no se generó.")
    end

    rules  = get(cfg, "rules", Dict())
    agents = get(cfg, "agents", Dict())
    pop    = get(cfg, "population", Dict())
    viz    = get(cfg, "visualization", Dict())

    if category == "grid_discrete"
        ag = get(rules, "agent_step", "")
        if !isempty(ag) && !(ag in BUILTIN_RULES)
            occursin(Regex("function\\s+\\Q" * ag * "\\E\\s*\\("), jl_str) ||
                return (false, "agent_step=\"$ag\" no es built-in y no está definido en el .jl.")
        end
        if !isempty(jl_str) && (occursin("@agent", jl_str) || occursin(r"^\s*using\b"m, jl_str))
            return (false, "En grid_discrete el .jl no debe contener @agent ni using.")
        end
        # The grid must be fully populated or the automaton freezes on frame 1.
        if haskey(pop, "pop_density")
            dens = values(pop["pop_density"])
            total = sum(Float64.(collect(dens)); init = 0.0)
            isapprox(total, 1.0; atol = 0.02) ||
                return (false, "pop_density debe sumar 1.0 para llenar todo el grid (suma $total). " *
                               "Para Bool usa { true = 0.3, false = 0.7 }.")
        end

    elseif category == "continuous_field"
        get(agents, "state_type", "") == "Float64" ||
            return (false, "continuous_field requiere state_type=\"Float64\".")

        # model_step: built-in (lenia_model_step! puro | lenia_noisy_step! con ruido) o uno
        # propio definido en el _rules.jl.
        ms = get(rules, "model_step", "")
        isempty(ms) &&
            return (false, "continuous_field requiere rules.model_step (\"lenia_model_step!\" o \"lenia_noisy_step!\").")
        if !(ms in BUILTIN_RULES)
            occursin(Regex("function\\s+\\Q" * ms * "\\E\\s*\\("), jl_str) ||
                return (false, "model_step=\"$ms\" no existe. NO renombres la función al tema del " *
                               "organismo: usa EXACTAMENTE \"lenia_model_step!\" (o \"lenia_noisy_step!\" " *
                               "si hay ruido). Solo define una función propia en el _rules.jl si vas a diseñar una regla nueva.")
        end

        # post_init: built-in (lenia_init!) o uno propio (p.ej. uno que diseñe la función de
        # crecimiento G fijando abmproperties(model)[:growth_fn]).
        pii = get(rules, "post_init", "")
        isempty(pii) &&
            return (false, "continuous_field requiere rules.post_init (p.ej. \"lenia_init!\").")
        if !(pii in BUILTIN_RULES)
            occursin(Regex("function\\s+\\Q" * pii * "\\E\\s*\\("), jl_str) ||
                return (false, "post_init=\"$pii\" no existe. NO lo renombres al tema del organismo: " *
                               "usa EXACTAMENTE \"lenia_init!\". Solo define un post_init propio en el " *
                               "_rules.jl si vas a DISEÑAR una G/kernel nuevos.")
        end

        # Si model_step o post_init son personalizados, hace falta el _rules.jl.
        if (!(ms in BUILTIN_RULES) || !(pii in BUILTIN_RULES)) && isempty(jl_str)
            return (false, "Un model_step/post_init a medida (p.ej. una G diseñada) requiere un _rules.jl.")
        end

        props = get(cfg, "properties", Dict())

        # Si se usan los steps built-in, las propiedades deben llamarse lenia_mu/lenia_sigma:
        # lenia_model_step! las lee por esos nombres exactos (model.lenia_mu), así que un
        # renombrado al tema (organism_mu, …) compila pero peta en ejecución. Lo cazamos aquí.
        if ms in ("lenia_model_step!", "lenia_noisy_step!")
            for pk in ("lenia_mu", "lenia_sigma")
                haskey(props, pk) ||
                    return (false, "Falta la propiedad [properties].$pk. lenia_model_step! la lee por " *
                                   "ese nombre EXACTO; no la renombres al tema (p.ej. organism_mu/creature_sigma son inválidas).")
            end
        end

        # Una semilla de criatura (lenia_orbium, …) necesita una rejilla amplia para desplazarse.
        init_rule = get(rules, "initialization_rule", "")
        if startswith(init_rule, "lenia_") && init_rule != "uniform_float"
            dims = get(get(cfg, "space", Dict()), "dimensions", Any[])
            if dims isa AbstractVector && length(dims) == 2 && all(d -> d isa Integer, dims)
                minimum(dims) >= 64 ||
                    return (false, "initialization_rule=\"$init_rule\" siembra una criatura que necesita " *
                                   "sitio para desplazarse: usa [space].dimensions de al menos [64, 64] (mejor [128, 128]).")
            end
        end

        # Parámetros de perturbación (opcionales): validar tipo/rango si aparecen.
        if haskey(props, "sigma_noise")
            sn = props["sigma_noise"]
            (sn isa Real && sn >= 0) ||
                return (false, "sigma_noise debe ser un número ≥ 0 (es $(repr(sn))).")
        end
        if haskey(props, "noise_scope")
            props["noise_scope"] in ("support", "field") ||
                return (false, "noise_scope debe ser \"support\" o \"field\".")
        end
        for nk in ("noise_start_step", "noise_end_step")
            if haskey(props, nk)
                (props[nk] isa Integer && props[nk] >= 0) ||
                    return (false, "$nk debe ser un entero ≥ 0.")
            end
        end

        # [run] (run!): las métricas de modelo deben ir en mdata, no en adata.
        runc = get(cfg, "run", Dict())
        if !isempty(runc)
            ad = get(runc, "adata", Any[])
            misplaced = [m for m in ad if m in MODEL_METRICS]
            isempty(misplaced) ||
                return (false, "Las métricas de modelo $(misplaced) deben ir en [run].mdata, NO en adata. " *
                               "Pon mdata=[\"total_energy\",\"active_cells\"] y deja adata=[].")
        end

    elseif category == "continuous_space"
        (occursin("@agent", jl_str) && occursin("using Agents", jl_str)) ||
            return (false, "continuous_space requiere un .jl con @agent y using Agents.")
        haskey(pop, "pop_quantity") ||
            return (false, "continuous_space requiere population.pop_quantity.")

        # Cada campo `const` del struct @agent se instancia leyéndolo de [agents] por nombre
        # (ver populate_continuous_world!). id/pos/vel los inyecta @agent automáticamente.
        # Comprobamos la correspondencia aquí para que el bucle de reparación reciba un
        # mensaje preciso en vez de fallar solo en ejecución.
        struct_fields = [m.captures[1] for m in eachmatch(r"const\s+(\w+)\s*::", jl_str)]
        missing_fields = [f for f in struct_fields if !haskey(agents, f)]
        isempty(missing_fields) ||
            return (false, "Cada campo `const` del struct @agent debe tener un valor en [agents] del .toml. " *
                           "Faltan: $(join(missing_fields, ", ")). Añádelos bajo [agents] (NO declares id/pos/vel).")

    elseif category == "hexagonal"
        get(viz, "agent_shape", "") == "hexagon" ||
            return (false, "hexagonal requiere agent_shape=\"hexagon\".")
    end

    return (true, "")
end

# --- Response parser (FILENAME blocks / language fallback) ---

function parse_llm_response(text::AbstractString)
    clean = replace(String(text), r"\e\[[0-9;]*m" => "")
    clean = replace(clean, "\r\n" => "\n")

    files = _parse_filename_blocks(clean)
    !isempty(files) && return files
    return _parse_by_language(clean)
end

function _parse_filename_blocks(text::AbstractString)
    pattern = r"FILENAME\s*:\s*([a-zA-Z0-9_\.\-]+)[^\n]*\n\s*```[a-zA-Z]*\n(.*?)\n```"s
    files   = Dict{String,String}()
    for m in eachmatch(pattern, text)
        content = strip(m.captures[2])
        # Skip the placeholder template from the system prompt's OUTPUT FORMAT section
        # (it gets echoed back by the model in some runs).
        (isempty(content) || content == "..." ||
            occursin("toml content", content) || occursin("julia content", content)) && continue
        files[strip(m.captures[1])] = content
    end
    return files
end

function _parse_by_language(text::AbstractString)
    files = Dict{String,String}()
    tm = match(r"```toml\n(.*?)\n```"s, text)
    jm = match(r"```(?:julia|jl)\n(.*?)\n```"s, text)
    isnothing(tm) && isnothing(jm) && return files

    base = "simulation"
    if !isnothing(tm)
        nm = match(r"model_name\s*=\s*\"([^\"]+)\"", tm.captures[1])
        if !isnothing(nm)
            base = lowercase(replace(strip(nm.captures[1]), r"[\s\-]+" => "_"))
        end
        files["$base.toml"] = strip(tm.captures[1])
    end
    if !isnothing(jm)
        files["$(base)_rules.jl"] = strip(jm.captures[1])
    end
    return files
end

function _dump_raw(response::AbstractString, output_dir::String)
    path = abspath(joinpath(output_dir, "llm_raw_output.txt"))
    write(path, response)
    println("Salida cruda del LLM guardada para inspección: $path")
end

# --- Generation + write + run ---

function _generate(category::String, desc::String; feedback::String = "")
    system = load_prompt(category)
    user   = isempty(feedback) ? desc : "$desc\n\n$feedback"
    resp   = llm_call(system, user; n_predict = 4096)
    return parse_llm_response(resp), resp
end

function _write_files(files::Dict, output_dir::String)
    toml_path = ""
    jl_path   = ""
    println("\nArchivos generados en: $(abspath(output_dir))")
    for (fn, content) in files
        p = abspath(joinpath(output_dir, fn))
        write(p, content * "\n")
        println("    $p")
        endswith(fn, ".toml") && (toml_path = p)
        endswith(fn, ".jl")   && (jl_path   = p)
    end
    return (toml_path, jl_path)
end

# Runs the generated simulation in a subprocess (so rules load cleanly).
# Returns (ok::Bool, stderr::String).
function _run_simulation(toml_path::String, jl_path::String)
    main_script = abspath(joinpath(@__DIR__, "..", "examples", "main.jl"))
    project_dir = abspath(joinpath(@__DIR__, ".."))
    cmd = isempty(jl_path) ?
        `julia --project=$project_dir $main_script $toml_path` :
        `julia --project=$project_dir $main_script $toml_path $jl_path`
    out  = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    output = String(take!(out))
    return (success(proc), output)
end

# --- Main entry point ---

"""
    build_from_prompt(user_text; interactive=true, output_dir=...) -> String

Traduce una descripción en lenguaje natural (`user_text`) a una simulación de Cell_IA
ejecutable, usando el LLM local. Enruta la petición a una categoría de modelo, genera el
`.toml` (y, si hace falta, un `_rules.jl`), valida la salida con un bucle de reparación y la
escribe en `output_dir`. Con `interactive=true` pide confirmación antes de generar.
Devuelve la ruta `output_dir` con los archivos generados.
"""
function build_from_prompt(user_text::String;
                           interactive::Bool = true,
                           output_dir = joinpath(@__DIR__, "..", "examples"))
    mkpath(output_dir)

    # 1+2. Route and confirm.
    sel = confirm_and_select(user_text; interactive = interactive)
    sel === nothing && (println("Cancelado."); return output_dir)
    category, desc = sel

    # 3. Generate with a static-validation repair loop.
    files = Dict{String,String}()
    raw   = ""
    feedback = ""
    valid = false
    for attempt in 1:3
        files, raw = _generate(category, desc; feedback = feedback)
        if isempty(files)
            feedback = "No emitiste los archivos en el formato FILENAME: + bloque de código. Repítelo EXACTAMENTE en ese formato."
            continue
        end
        ok, msg = validate_files(files, category)
        if ok
            valid = true
            break
        end
        @warn "Validación falló (intento $attempt): $msg"
        feedback = "El intento anterior falló la validación: $msg. Corrige SOLO eso y vuelve a emitir los archivos completos en el formato FILENAME:."
    end

    if isempty(files)
        @error "El LLM no devolvió archivos en el formato esperado."
        _dump_raw(raw, output_dir)
        return output_dir
    end
    valid || @warn "Se generaron archivos pero la validación estática no quedó limpia; se intentará ejecutar igualmente."

    # 4. Write + run with an execution-error repair loop.
    toml_path, jl_path = _write_files(files, output_dir)
    for attempt in 1:2
        println("\nEjecutando simulación (intento $attempt)...")
        ok, errmsg = _run_simulation(toml_path, jl_path)
        if ok
            println("Simulación completada.")
            return output_dir
        end
        @warn "La ejecución falló."
        println(errmsg)
        if attempt < 2
            fb = "Al ejecutar, Julia dio este error:\n$errmsg\n\nCorrige los archivos para que ejecute sin error. Vuelve a emitir AMBOS archivos completos en el formato FILENAME:."
            files, raw = _generate(category, desc; feedback = fb)
            isempty(files) && break
            toml_path, jl_path = _write_files(files, output_dir)
        end
    end

    println("No se pudo ejecutar sin errores automáticamente. Revisa los archivos y ejecuta a mano:")
    println("    julia --project=. examples/main.jl $toml_path" *
            (isempty(jl_path) ? "" : " $jl_path"))
    return output_dir
end

end
