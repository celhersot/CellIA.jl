module LLMBuilder
using LlamaCpp
using Downloads
using TOML

export build_from_prompt

const MODEL_DIR = joinpath(@__DIR__, "..", "models")
const MODEL_NAME = "qwen2.5-1.5b-coder-q4_k_m.gguf"
const MODEL_PATH = joinpath(MODEL_DIR, MODEL_NAME)
const MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
#const SYSTEM_PROMPT_PATH = joinpath(@__DIR__, "system_prompt.txt")
const SYSTEM_PROMPT_PATH = joinpath(@__DIR__, "dummy_prompt.txt")
const SYSTEM_PROMPT = read(SYSTEM_PROMPT_PATH, String)

function ensure_model_exists()
    if !isdir(MODEL_DIR)
        println("--> Using downloaded model")
        mkpath(MODEL_DIR)
    end

    if !isfile(MODEL_PATH)
        println("--> 📦 First execution: Downloading brain...")
        println("--> This will only happen once. Please wait...")
        try
            Downloads.download(MODEL_URL, MODEL_PATH)
            println("--> Model downloaded successfully.")
        catch e
            error("Error downloading the template. Check your internet connection: $e")
        end
    end
    return MODEL_PATH
end

function build_from_prompt(user_text::String; output_dir=joinpath(@__DIR__, "..", "examples2"))
    model_path = ensure_model_exists()
    mkpath(output_dir)
    
    println("--> Local LLM is thinking (this might take a moment)...")
    response_text = run_llama(model_path, user_text)
    println("FINISH")
    
    
    #clean_text = replace(response_text, r"\e\[[0-9;]*m" => "")
    #clean_text = replace(clean_text, r".*available commands:.*?(\n> )?"s => "")
    #clean_text = replace(clean_text, r"\[ Prompt: .*? \]" => "") # Quita las estadísticas finales

    #pattern = r"FILENAME:\s*([a-zA-Z0-9_\.-]+)\s*```[a-zA-Z]*\n(.*?)\n```"s
    
    # found_any = false
    # toml_path = ""
    # jl_path = ""
    
    # for m in eachmatch(pattern, clean_text)
    #     filename = m.captures[1]
    #     content = m.captures[2]
        
    #     file_path = abspath(joinpath(output_dir, filename))
        
    #     open(file_path, "w") do f
    #         write(f, content)
    #     end
        
    #     println("--> [OK] Saved: $file_path")
    #     found_any = true

    #     if endswith(filename, ".toml")
    #         toml_path = file_path
    #     elseif endswith(filename, ".jl")
    #         jl_path = file_path
    #     end
    # end

    # explanation = clean_text
    # for m in eachmatch(pattern, clean_text)
    #     explanation = replace(explanation, m.match => "")
    # end
    
    # explanation = replace(explanation, r"<\|im_start\|>.*?(<\|im_end\|>|$)"s => "")
    # explanation = strip(explanation)

    # if !isempty(explanation)
    #     println("\n--- LLM Decision-Making Explanation ---")
    #     println(explanation)
    #     println("------------------------------\n")
    # end

    # if !found_any
    #     @warn "The LLM did not return code blocks in the expected FILENAME format."
    #     return output_dir
    # end

    # if !isempty(toml_path)
    #     println("Executing generated simulation...")
        
    #     main_script = isfile("main.jl") ? "main.jl" : abspath(joinpath(@__DIR__, "..", "main.jl"))
        
    #     cmd_args = isempty(jl_path) ? `julia $main_script $toml_path` : `julia $main_script $toml_path $jl_path`
        
    #     try
    #         run(cmd_args)
    #         println("END OF SIMULATION.")
            
    #         config = TOML.parsefile(toml_path)
    #         if haskey(config, "visualization") && haskey(config["visualization"], "filename")
    #             vid_file = config["visualization"]["filename"]
    #             vid_path = abspath(vid_file)
                
    #             if isfile(vid_path)
    #                 println("--> Playing video: $vid_path")
    #                 if Sys.iswindows()
    #                     run(`cmd /c start "" "$vid_path"`)
    #                 elseif Sys.isapple()
    #                     run(`open "$vid_path"`)
    #                 elseif Sys.islinux()
    #                     run(`xdg-open "$vid_path"`)
    #                 end
    #             else
    #                 @warn "The generated video was not found on the expected route: $vid_path"
    #             end
    #         end
    #     catch e
    #         @error "ERROR: executing simulation or playing video: $e"
    #     end
    # end
    
    # return output_dir
    
end

function run_llama(model_path, user_description)
    prompt_file = abspath(joinpath(@__DIR__, "input_prompt.txt"))
    output_file = abspath(joinpath(@__DIR__, "output_llm.txt"))
    
    full_prompt = "<|im_start|>system\n$SYSTEM_PROMPT<|im_end|>\n<|im_start|>user\n$user_description<|im_end|>\n<|im_start|>assistant\n"
    write(prompt_file, full_prompt)
    
    llama_exe = abspath(joinpath(@__DIR__, "..", "bin", "llama-cli.exe"))
    model_abs = abspath(model_path)
    
    # try
    println("Generating simulation...")
    cmd = `$llama_exe --model $model_abs -f $prompt_file --n-gpu-layers 99 --ctx-size 8192 --n-predict 4096 --no-display-prompt --log-disable`
    println("después de cmd")
    run(cmd)
    #run(pipeline(cmd, stdin=devnull, stdout=output_file, stderr=devnull))
    println("después de run")

    if isfile(output_file)
        response = read(output_file, String)
        println("Response saved.")
        return response
    end
    # catch e
    #     @error "The execution failed: $e"
    #     return ""
    # finally
    #     isfile(prompt_file) && rm(prompt_file, force=true)
    #     isfile(output_file) && rm(output_file, force=true)
    # end
    # return ""
end

 #build_from_prompt("basic plague simulation with death, sick and healthy agents")
 build_from_prompt("Dime hola y adios")

end