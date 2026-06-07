exe   = joinpath(@__DIR__, "bin", "llama-completion.exe")
model = joinpath(@__DIR__, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
prompt = "<|im_start|>user\nCount from 1 to 30, comma separated.<|im_end|>\n<|im_start|>assistant\n"
N = 64
cmd = `$exe --model $model -p $prompt --n-gpu-layers 99 --ctx-size 1024 --n-predict $N --temp 0.1 --no-display-prompt --threads 4`
out = IOBuffer(); err = IOBuffer()
# stdin=devnull => if it enters interactive mode it gets EOF and exits immediately.
t = @elapsed run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = err))
println("=== total $(round(t, digits=1)) s  =>  $(round(N/t, digits=2)) tok/s (incluye carga del modelo) ===")
e = String(take!(err))
for ln in split(e, "\n")
    if occursin(r"(?i)offload|n_gpu|tokens per second|eval time|load time|prompt eval|vulkan|error|out of|CPU_Mapped", ln)
        println(ln)
    end
end
println("--- output ---")
println(strip(String(take!(out))))
println("PROBE5_DONE")
