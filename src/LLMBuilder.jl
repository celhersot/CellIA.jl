module LLMBuilder
using Downloads
using TOML

export build_from_prompt

const MODEL_DIR  = joinpath(@__DIR__, "..", "models")
const MODEL_NAME = "qwen2.5-1.5b-coder-q4_k_m.gguf"
const MODEL_PATH = joinpath(MODEL_DIR, MODEL_NAME)
const MODEL_URL  = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
const SYSTEM_PROMPT_PATH = joinpath(@__DIR__, "system_prompt.txt")
const SYSTEM_PROMPT = read(SYSTEM_PROMPT_PATH, String)

# ── Model download ─────────────────────────────────────────────────────────────

function ensure_model_exists()
    if !isdir(MODEL_DIR)
        mkpath(MODEL_DIR)
    end
    if !isfile(MODEL_PATH)
        println("--> First execution: downloading model (one-time only)...")
        try
            Downloads.download(MODEL_URL, MODEL_PATH)
            println("--> Model downloaded.")
        catch e
            error("Failed to download model. Check internet connection: $e")
        end
    end
    return MODEL_PATH
end

# ── LLM call ──────────────────────────────────────────────────────────────────
# Writes the prompt to a temp file, calls llama-cli, captures stdout,
# and always deletes the temp file when done.

function run_llama(model_path::String, user_description::String)::String
    prompt_file = abspath(joinpath(@__DIR__, "input_prompt.txt"))
    output_file = abspath(joinpath(@__DIR__, "output_llm.txt"))
    full_prompt  = "<|im_start|>system\n$SYSTEM_PROMPT<|im_end|>\n" *
                   "<|im_start|>user\n$user_description<|im_end|>\n" *
                   "<|im_start|>assistant\n"
    try
        write(prompt_file, full_prompt)
        llama_exe = abspath(joinpath(@__DIR__, "..", "bin", "llama-cli.exe"))
        cmd = `$llama_exe --model $(abspath(model_path)) -f $prompt_file --n-gpu-layers 99 --ctx-size 8192 --n-predict 4096 --no-display-prompt --log-disable`
        println("--> Local LLM thinking (can take several minutes on CPU)...")
        run(pipeline(cmd; stdin=devnull, stdout=output_file, stderr=devnull))
        isfile(output_file) || return ""
        return read(output_file, String)
    catch e
        @error "LLM execution failed: $e"
        return ""
    finally
        try; isfile(prompt_file) && rm(prompt_file; force=true); catch; end
        try; isfile(output_file) && rm(output_file; force=true); catch; end
    end
end

# ── Response parser ────────────────────────────────────────────────────────────
# Extracts FILENAME / ``` blocks from LLM output.
# Returns a Dict mapping filename => content string.

function parse_llm_response(text::String)
    clean = replace(text, r"\e\[[0-9;]*m" => "")   # strip ANSI colours
    clean = replace(clean, "\r\n" => "\n")           # normalise Windows line endings

    # Extract only the assistant turn — everything between
    # <|im_start|>assistant\n and the llama-cli stats line [ Prompt: ... ]
    m = match(r"<\|im_start\|>assistant\n(.*?)(?:\[[ \d\.]+t/s[^\]]*\]|Exiting\.\.\.|$)"s, clean)
    clean = m !== nothing ? strip(m.captures[1]) : clean

    # Strategy 1 — explicit FILENAME: blocks with real content (not "...")
    files = _parse_filename_blocks(clean)
    !isempty(files) && return files

    # Strategy 2 — extract by language tag (```toml / ```julia) and derive
    # filenames from the model_name field inside the TOML block.
    _parse_by_language(clean)
end

function _parse_filename_blocks(text::String)
    pattern = r"FILENAME\s*:\s*([a-zA-Z0-9_\.\-]+)[^\n]*\n\s*```[a-zA-Z]*\n(.*?)\n```"s
    files   = Dict{String,String}()
    for m in eachmatch(pattern, text)
        content = strip(m.captures[2])
        (isempty(content) || content == "...") && continue
        files[strip(m.captures[1])] = content
    end
    return files
end

function _parse_by_language(text::String)
    files = Dict{String,String}()
    tm = match(r"```toml\n(.*?)\n```"s, text)
    jm = match(r"```(?:julia|jl)\n(.*?)\n```"s, text)
    isnothing(tm) && isnothing(jm) && return files

    # Derive a safe base name from model_name inside the TOML, or fall back.
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

function _dump_raw(response::String, output_dir::String)
    path = abspath(joinpath(output_dir, "llm_raw_output.txt"))
    write(path, response)
    println("--> Raw LLM output saved for inspection: $path")
end

# ── Main entry point ───────────────────────────────────────────────────────────
# 1. Call LLM, parse response, save .toml and .jl files.
# 2. Automatically run the simulation (subprocess so rules load cleanly).
# 3. Temp files (input_prompt.txt) are deleted in run_llama's finally block.

function build_from_prompt(user_text::String;
                           output_dir=joinpath(@__DIR__, "..", "examples"))
    model_path = ensure_model_exists()
    mkpath(output_dir)

    response = run_llama(model_path, user_text)

    if isempty(response)
        @error "LLM returned an empty response."
        return output_dir
    end

    files = parse_llm_response(response)

    if isempty(files)
        @warn "LLM did not return FILENAME blocks in the expected format."
        _dump_raw(response, output_dir)
        return output_dir
    end

    toml_path = ""
    jl_path   = ""

    println("\n--> Generated files saved in: $(abspath(output_dir))")
    for (filename, content) in files
        file_path = abspath(joinpath(output_dir, filename))
        write(file_path, content * "\n")
        println("    ✓ $file_path")
        if endswith(filename, ".toml")
            toml_path = file_path
        elseif endswith(filename, ".jl")
            jl_path = file_path
        end
    end

    # Print any explanation text the LLM added after the code blocks
    clean   = replace(response, r"\e\[[0-9;]*m" => "")
    pattern = r"FILENAME:\s*([a-zA-Z0-9_\.\-]+)\s*```[a-zA-Z]*\n(.*?)\n```"s
    explanation = clean
    for m in eachmatch(pattern, clean)
        explanation = replace(explanation, m.match => "")
    end
    explanation = strip(replace(explanation, r"<\|im_start\|>.*?(<\|im_end\|>|$)"s => ""))
    if !isempty(explanation)
        println("\n--- LLM explanation ---")
        println(explanation)
        println("-----------------------\n")
    end

    # Auto-run the generated simulation
    if !isempty(toml_path)
        main_script  = abspath(joinpath(@__DIR__, "..", "examples", "main.jl"))
        project_dir  = abspath(joinpath(@__DIR__, ".."))
        cmd = isempty(jl_path) ?
            `julia --project=$project_dir $main_script $toml_path` :
            `julia --project=$project_dir $main_script $toml_path $jl_path`

        println("--> Running simulation...")
        try
            run(cmd)
            println("--> Simulation complete.")
        catch e
            @error "Simulation failed: $e"
            println("--> Check the generated files for errors and re-run manually:")
            println("    julia --project=. examples/main.jl $toml_path" *
                    (isempty(jl_path) ? "" : " $jl_path"))
        end
    end

    return output_dir
end

end
