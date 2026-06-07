module LLMBuilder

# Local-LLM driven generator for Cell_IA simulations.
#
# Flow (see SPEC_MultiPrompt.md):
#   1. route()                -> router-agent classifies the request into one of 4 categories
#   2. confirm (CLI)          -> user accepts / rejects / corrects
#   3. generate + validate    -> per-category system prompt, static validation, repair loop
#   4. run + repair-on-error  -> execute; on Julia error, feed stderr back and regenerate
#
# Backend: llama-server over HTTP (confirmed). llama-cli is an emergency fallback.

using Downloads
using TOML
using HTTP
using JSON

export build_from_prompt, route, validate_files, load_prompt

# ── Model / server configuration ────────────────────────────────────────────────

const MODEL_DIR  = joinpath(@__DIR__, "..", "models")
const MODEL_NAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
const MODEL_PATH = joinpath(MODEL_DIR, MODEL_NAME)
# 1.5B: the model that already worked (~3 min/gen) on this machine. ~1 GB, fits in
# the MX130's 2 GB VRAM so it runs on the GPU. (A 7B was a mistake — too big here.)
const MODEL_URL  = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

const SERVER_HOST = get(ENV, "LLAMA_SERVER_HOST", "127.0.0.1")
const SERVER_PORT = get(ENV, "LLAMA_SERVER_PORT", "8080")

const PROMPTS_DIR = joinpath(@__DIR__, "prompts")

# category => (prompt file, expected space.type, whether a _rules.jl is mandatory)
const CATEGORIES = Dict(
    "grid_discrete"    => (prompt = "grid_discrete.txt",    space = "grid",       needs_jl = false),
    "continuous_field" => (prompt = "continuous_field.txt", space = "grid",       needs_jl = false),
    "continuous_space" => (prompt = "continuous_space.txt", space = "continuous", needs_jl = true),
    "hexagonal"        => (prompt = "hexagonal.txt",         space = "hexagonal",  needs_jl = true),
)

const BUILTIN_RULES = Set([
    "gol_step!", "rps_step!", "schelling_step!", "default_model_step!",
    "lenia_model_step!", "lenia_init!",
])

# Read once at load time (the files ship with the framework).
const ROUTER_GRAMMAR = read(joinpath(PROMPTS_DIR, "router.gbnf"), String)

# ── Prompt loading ───────────────────────────────────────────────────────────────

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

# ── Model download (only needed for the llama-cli fallback) ──────────────────────

function ensure_model_exists()
    isdir(MODEL_DIR) || mkpath(MODEL_DIR)
    if !isfile(MODEL_PATH)
        println("--> Descargando modelo (una sola vez, ~9 GB)...")
        try
            Downloads.download(MODEL_URL, MODEL_PATH)
            println("--> Modelo descargado.")
        catch e
            error("No se pudo descargar el modelo. Revisa la conexión: $e")
        end
    end
    return MODEL_PATH
end

# ── LLM call ─────────────────────────────────────────────────────────────────────
# Builds a ChatML prompt and sends it to llama-server (/completion). Falls back to
# llama-cli if the server is unreachable. Returns the assistant text.

function llm_call(system::String, user::String;
                  grammar::Union{Nothing,String} = nothing,
                  n_predict::Int = 4096,
                  temperature::Float64 = 0.2)
    prompt = "<|im_start|>system\n$system<|im_end|>\n" *
             "<|im_start|>user\n$user<|im_end|>\n" *
             "<|im_start|>assistant\n"
    # One-shot CLI call, no server: load model -> generate -> exit. Same simple
    # approach as your original setup, just split across router + generation calls.
    return _cli_completion(prompt; grammar = grammar, n_predict = n_predict)
end

function _server_completion(prompt::String; grammar, n_predict, temperature)
    body = Dict{String,Any}(
        "prompt"       => prompt,
        "n_predict"    => n_predict,
        "temperature"  => temperature,
        "stop"         => ["<|im_end|>"],
        "cache_prompt" => true,
    )
    grammar !== nothing && (body["grammar"] = grammar)
    url  = "http://$(SERVER_HOST):$(SERVER_PORT)/completion"
    resp = HTTP.post(url, ["Content-Type" => "application/json"], JSON.json(body);
                     connect_timeout = 3, readtimeout = 900, retry = false)
    data = JSON.parse(String(resp.body))
    return strip(get(data, "content", ""))
end

function _cli_completion(prompt::String; grammar, n_predict)
    model_path   = ensure_model_exists()
    exe_name     = Sys.iswindows() ? "llama-completion.exe" : "llama-completion"
    llama_exe    = abspath(joinpath(@__DIR__, "..", "bin", exe_name))
    grammar_file = abspath(joinpath(@__DIR__, "grammar.gbnf"))
    # -p (prompt string) => one-shot. --n-gpu-layers 99 => use the GPU (the 1.5B fits
    # in 2 GB VRAM). stdin=devnull => if it tried to go interactive it gets EOF and exits.
    cmd = `$llama_exe --model $(abspath(model_path)) -p $prompt --n-gpu-layers 99 --ctx-size 8192 --n-predict $n_predict --temp 0.2 --no-display-prompt`
    if grammar !== nothing
        write(grammar_file, grammar)
        cmd = `$cmd --grammar-file $grammar_file`
    end
    out = IOBuffer()
    try
        run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = devnull))
        return strip(String(take!(out)))
    catch e
        @error "Ejecución del modelo falló: $e"
        return ""
    finally
        try; isfile(grammar_file) && rm(grammar_file; force = true); catch; end
    end
end

# ── Router ──────────────────────────────────────────────────────────────────────

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
        raw = llm_call(system, desc; grammar = ROUTER_GRAMMAR, n_predict = 256, temperature = 0.1)
        r = parse_router_json(raw)
        r !== nothing && return r
    end
    error("El router no devolvió un JSON válido tras 2 intentos.")
end

# ── Confirmation loop (CLI) ───────────────────────────────────────────────────────
# Returns (category, final_description) or nothing if cancelled.

function confirm_and_select(user_text::String; interactive::Bool = true)
    excluded = String[]
    desc = user_text
    for _ in 1:6
        r = route(desc; exclude = excluded)
        if !interactive
            println("--> Categoría elegida automáticamente: $(r.category)")
            return (r.category, desc)
        end
        println("\n--> Propuesta: $(r.approach)")
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
            desc = desc * " " * ans          # treat as a correction; re-route
        end
    end
    println("Demasiados intentos sin confirmar. Cancelando.")
    return nothing
end

# ── Static validation ─────────────────────────────────────────────────────────────
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

    elseif category == "continuous_field"
        get(agents, "state_type", "") == "Float64" ||
            return (false, "continuous_field requiere state_type=\"Float64\".")
        get(rules, "model_step", "") == "lenia_model_step!" ||
            return (false, "continuous_field requiere model_step=\"lenia_model_step!\".")
        haskey(rules, "post_init") ||
            return (false, "continuous_field requiere rules.post_init (p.ej. \"lenia_init!\").")

    elseif category == "continuous_space"
        (occursin("@agent", jl_str) && occursin("using Agents", jl_str)) ||
            return (false, "continuous_space requiere un .jl con @agent y using Agents.")
        haskey(pop, "pop_quantity") ||
            return (false, "continuous_space requiere population.pop_quantity.")

    elseif category == "hexagonal"
        get(viz, "agent_shape", "") == "hexagon" ||
            return (false, "hexagonal requiere agent_shape=\"hexagon\".")
    end

    return (true, "")
end

# ── Response parser (FILENAME blocks / language fallback) ─────────────────────────

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
        (isempty(content) || content == "...") && continue
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
    println("--> Salida cruda del LLM guardada para inspección: $path")
end

# ── Generation + write + run ──────────────────────────────────────────────────────

function _generate(category::String, desc::String; feedback::String = "")
    system = load_prompt(category)
    user   = isempty(feedback) ? desc : "$desc\n\n$feedback"
    resp   = llm_call(system, user; n_predict = 4096)
    return parse_llm_response(resp), resp
end

function _write_files(files::Dict, output_dir::String)
    toml_path = ""
    jl_path   = ""
    println("\n--> Archivos generados en: $(abspath(output_dir))")
    for (fn, content) in files
        p = abspath(joinpath(output_dir, fn))
        write(p, content * "\n")
        println("    ✓ $p")
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

# ── Main entry point ───────────────────────────────────────────────────────────────

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
        println("\n--> Ejecutando simulación (intento $attempt)...")
        ok, errmsg = _run_simulation(toml_path, jl_path)
        if ok
            println("--> ✓ Simulación completada.")
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

    println("--> No se pudo ejecutar sin errores automáticamente. Revisa los archivos y ejecuta a mano:")
    println("    julia --project=. examples/main.jl $toml_path" *
            (isempty(jl_path) ? "" : " $jl_path"))
    return output_dir
end

end
