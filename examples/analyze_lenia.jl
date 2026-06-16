# Análisis del experimento de resiliencia de Lenia (post-simulación).
#
# Específico del experimento (no es parte del framework). Lee el CSV de métricas que produce
# run_simulation, captura la línea base de energía previa a la perturbación, evalúa la ventana
# de recuperación y emite un VEREDICTO: SUPERVIVENCIA / DESINTEGRACIÓN / SATURACIÓN.
# Genera además la gráfica E(t) y celdas activas con el paso de perturbación marcado.
#
# Uso:
#   julia --project=. examples/analyze_lenia.jl [config.toml] [metrics.csv]
# Por defecto: examples/lenia_perturbation.toml  y  output_data/lenia_perturbation.csv

using TOML
using CairoMakie
using Statistics

const CFG_PATH = length(ARGS) >= 1 ? ARGS[1] : "examples/lenia_perturbation.toml"
const CSV_PATH = length(ARGS) >= 2 ? ARGS[2] : "output_data/lenia_perturbation.csv"

# ── Lectura del CSV simple (cabecera + filas separadas por comas) ───────────────────────
function read_csv(path)
    lines = readlines(path)
    @assert !isempty(lines) "CSV vacío: $path"
    header = String.(split(strip(lines[1]), ","))
    cols = Dict(h => Float64[] for h in header)
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        vals = split(strip(ln), ",")
        length(vals) == length(header) || continue
        for (h, v) in zip(header, vals)
            push!(cols[h], something(tryparse(Float64, v), NaN))
        end
    end
    return header, cols
end

# Localiza una columna por cualquiera de sus nombres posibles.
function pick(cols, names...)
    for n in names
        haskey(cols, n) && return cols[n]
    end
    error("No encuentro ninguna de las columnas $(names) en el CSV. Columnas: $(keys(cols))")
end

function main()
    cfg  = TOML.parsefile(CFG_PATH)
    dims = Tuple(cfg["space"]["dimensions"])
    N    = prod(dims)
    K    = Int(get(cfg["properties"], "noise_start_step", 0))
    σ    = Float64(get(cfg["properties"], "sigma_noise", 0.0))
    W    = 50                                   # ventana base/recuperación (D6)

    _, cols = read_csv(CSV_PATH)
    t = haskey(cols, "time") ? cols["time"] : haskey(cols, "step") ? cols["step"] : collect(0.0:length(first(values(cols)))-1)
    E = pick(cols, "total_energy")
    A = pick(cols, "active_cells")
    n = length(E)

    # Índices de las ventanas (los pasos del CSV empiezan en t=0).
    base_lo = max(1, K - W); base_hi = max(base_lo, K - 1)
    base_E  = K > 1 ? mean(@view E[base_lo:min(base_hi, n)]) : E[1]
    base_A  = K > 1 ? mean(@view A[base_lo:min(base_hi, n)]) : A[1]
    fin_lo  = max(1, n - W + 1)
    final_E = mean(@view E[fin_lo:n])
    final_A = mean(@view A[fin_lo:n])

    # Umbrales del veredicto (D6).
    dis_thr = 0.05 * base_E            # energía cae por debajo del 5% de la base ⇒ disipación
    sat_thr = 0.50 * N                 # energía supera medio campo lleno ⇒ saturación
    rec_tol = 0.25                     # ±25% de la base ⇒ se considera recuperada

    verdict, detail = if final_E < dis_thr
        ("DESINTEGRACION (colapso)", "la energia se disipo hacia 0 (E_final=$(round(final_E;digits=2)) < $(round(dis_thr;digits=2)))")
    elseif final_E > sat_thr
        ("SATURACION (colapso)", "el campo se saturo hacia 1 (E_final=$(round(final_E;digits=2)) > $(round(sat_thr;digits=2)))")
    elseif abs(final_E - base_E) <= rec_tol * base_E
        ("SUPERVIVENCIA (robustez)", "la estructura absorbio el ruido y conservo su energia (E_base=$(round(base_E;digits=2)) -> E_final=$(round(final_E;digits=2)))")
    else
        ("DEFORMACION PARCIAL", "energia alterada pero sin colapsar (E_base=$(round(base_E;digits=2)) -> E_final=$(round(final_E;digits=2)))")
    end

    # Detección del paso de colapso (primera caída por debajo del 50% de las celdas activas base).
    collapse_step = nothing
    if K > 1
        for i in K:n
            if A[i] < 0.5 * base_A
                collapse_step = Int(round(t[i])); break
            end
        end
    end

    println("="^64)
    println("EXPERIMENTO: resiliencia de Lenia bajo perturbacion estocastica")
    println("="^64)
    println("  sigma_ruido        = $σ   (inyectado desde el paso K=$K)")
    println("  energia base       = $(round(base_E; digits=2))   (celdas activas base = $(round(base_A; digits=1)))")
    println("  energia final      = $(round(final_E; digits=2))   (celdas activas final = $(round(final_A; digits=1)))")
    isnothing(collapse_step) || println("  colapso detectado en el paso $collapse_step")
    println("-"^64)
    println("  VEREDICTO: $verdict")
    println("            $detail")
    println("="^64)

    # ── Gráfica: E(t) y celdas activas, con la perturbación marcada ─────────────────────
    fig = Figure(size = (900, 640))
    ax1 = Axis(fig[1, 1]; ylabel = "Energia total  E = Σ A", title = "Resiliencia de Lenia — $verdict")
    lines!(ax1, t, E; color = :dodgerblue, linewidth = 2, label = "E(t)")
    K > 1 && vlines!(ax1, [Float64(K)]; color = :red, linestyle = :dash, label = "perturbacion (K=$K)")
    K > 1 && hlines!(ax1, [base_E]; color = :gray, linestyle = :dot, label = "linea base")
    axislegend(ax1; position = :rb)

    ax2 = Axis(fig[2, 1]; xlabel = "paso", ylabel = "celdas activas")
    lines!(ax2, t, A; color = :seagreen, linewidth = 2)
    K > 1 && vlines!(ax2, [Float64(K)]; color = :red, linestyle = :dash)

    out = "output_photos/" * splitext(basename(CSV_PATH))[1] * "_energy.png"
    mkpath(dirname(out))
    save(out, fig)
    println("Grafica guardada: $out")
end

main()
