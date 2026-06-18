module CustomEvolutionRules
using Agents
using Random
using FFTW
using LinearAlgebra

# --- DEFAULT MODEL STEP ---

function default_model_step!(model)
    for agent in allagents(model)
        if haskey(model.next_states, agent.id)
            agent.state = model.next_states[agent.id]
        end
    end
    empty!(model.next_states)
end

# --- GAME OF LIFE ---

function gol_step!(agent, model)
    alives = count(n -> n.state == true, nearby_agents(agent, model))
    
    if agent.state == true
        model.next_states[agent.id] = (alives == model.min_to_live || alives == model.max_to_live)
    else
        model.next_states[agent.id] = (alives == model.max_to_live)
    end
end

# --- ROCK, PAPER, SCISSORS ---

function rps_step!(agent, model)
    beats = Dict(:rock => :paper, :paper => :scissors, :scissors => :rock)
    predator_type = beats[agent.state]
    
    n_predators = count(n -> n.state == predator_type, nearby_agents(agent, model))
    
    if n_predators >= model.threshold
        model.next_states[agent.id] = predator_type
    else
        model.next_states[agent.id] = agent.state
    end
end

### --- SCHELLING'S SEGREGATION MODEL ---

function schelling_step!(agent, model)
    neighbors = collect(nearby_agents(agent, model))
    
    # If alone -> happy
    if isempty(neighbors)
        return
    end

    same_group_count = count(n -> n.state == agent.state, neighbors)
    ratio = same_group_count / length(neighbors)
    
    # It moves
    if ratio < model.min_identical
        move_agent_single!(agent, model)
    end
end

# --- LENIA (automata celular continuo por convolucion FFT) ---
# Kernel y crecimiento gaussianos por defecto; se pueden sobreescribir por simulacion
# fijando abmproperties(model)[:kernel_fn] y/o [:growth_fn].

_lenia_kernel_fn(r::Float64) = r < 1.0 ? exp(4.0 - 1.0 / (r * (1.0 - r) + 1e-10)) : 0.0

_lenia_growth_fn(u::Float64, μ::Float64, σ::Float64) =
    2.0 * exp(-((u - μ)^2) / (2.0 * σ^2)) - 1.0

# Patrones semilla: estampados en un campo vacio dan un organismo coherente que planea.
# Se usan con initialization_rule="lenia_<name>" (p.ej. "lenia_orbium"); cada entrada trae
# sus parametros recomendados. Orbium (Bert Chan, 2019): planeador canonico, semilla 20x20.
const _ORBIUM_CELLS = [
[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
[0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
[0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
[0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
[0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
[0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
[0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
[0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
[0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
[0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
[0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
[0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
[0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
[0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
[0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
[0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
[0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
[0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
[0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
[0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0],
]
const ORBIUM = Float64.(permutedims(reduce(hcat, _ORBIUM_CELLS)))

# name => (seed pattern, recommended Lenia parameters for that creature).
const LENIA_CREATURES = Dict(
    "orbium" => (cells = ORBIUM, lenia_mu = 0.15, lenia_sigma = 0.017,
                 dt = 0.1, kernel_radius = 13, kernel_type = "gaussian"),
)

# Precomputa el kernel y el estado reutilizable del update (rfft del kernel, planes FFTW y
# buffers) para no replanificar ni reasignar en cada paso. Lo llama lenia_init!, o de forma
# perezosa el primer lenia_model_step!. :kernel_fft se mantiene por compatibilidad con legacy.
function _lenia_build_kernel!(model)
    dims = size(abmspace(model))
    R    = Int(get(abmproperties(model), :kernel_radius, 13))
    kfn  = get(abmproperties(model), :kernel_fn, _lenia_kernel_fn)

    K = zeros(Float64, dims...)
    for x in 1:dims[1], y in 1:dims[2]
        dx = min(x - 1, dims[1] - (x - 1))
        dy = min(y - 1, dims[2] - (y - 1))
        r  = sqrt(Float64(dx^2 + dy^2)) / R
        K[x, y] = kfn(r)
    end

    s = sum(K)
    s > 0.0 && (K ./= s)

    props = abmproperties(model)
    props[:kernel_fft] = fft(K)                          # compatibilidad con legacy

    # estado reutilizable: FFT real, planes y buffers
    Khat = rfft(K)                                       # rfft del kernel (una sola vez)
    A    = zeros(Float64, dims...)                       # buffer de campo (se rellena cada paso)
    Ahat = similar(Khat)                                 # buffer espectral
    U    = Matrix{Float64}(undef, dims...)               # buffer de convolución
    pf   = plan_rfft(A; flags = FFTW.MEASURE)            # r2c (planificar sobrescribe A; se rellena luego)
    pb   = plan_brfft(Ahat, dims[1]; flags = FFTW.MEASURE)  # c2r sin normalizar (la 1/N va aparte)
    props[:_lenia_state] = (A = A, U = U, Ahat = Ahat, Khat = Khat,
                            pf = pf, pb = pb, invN = 1.0 / (dims[1] * dims[2]))
    return nothing
end

# post_init por defecto: construye el kernel y el estado de trabajo.
function lenia_init!(model)
    kernel_type = string(get(abmproperties(model), :kernel_type, "gaussian"))
    if kernel_type == "polynomial"
        abmproperties(model)[:kernel_fn] = (r::Float64) -> max(0.0, 1.0 - r^2)^4
    end
    # kernel_type == "custom": el usuario ya fijo abmproperties(model)[:kernel_fn] antes
    _lenia_build_kernel!(model)
end

# Nucleo del paso en una funcion con tipos concretos: los buffers salen del Dict como Any,
# pero al pasarlos como argumentos el compilador especializa el bucle.
function _lenia_run_step!(model, st, μ::Float64, σ::Float64, dt::Float64, gfn::F) where {F}
    A, U, Ahat, Khat = st.A, st.U, st.Ahat, st.Khat
    pf, pb, invN     = st.pf, st.pb, st.invN

    # 1. snapshot agentes -> A (campo real)
    fill!(A, 0.0)
    @inbounds for a in allagents(model)
        A[a.pos[1], a.pos[2]] = a.state
    end

    # 2. convolución circular por FFT real:  U = irfft( rfft(A) .* rfft(K) )
    mul!(Ahat, pf, A)                       # Â = rfft(A)
    @inbounds @. Ahat = Ahat * Khat         # multiplicación en frecuencia (in-place)
    mul!(U, pb, Ahat)                       # U = brfft(Â)  (sin normalizar; la 1/N va en invN)

    # 3. crecimiento + clamp, fusionado e in-place sobre A
    @inbounds @. A = clamp(A + dt * gfn(U * invN, μ, σ), 0.0, 1.0)

    # 4. escribir el nuevo estado de vuelta a los agentes
    @inbounds for a in allagents(model)
        a.state = A[a.pos[1], a.pos[2]]
    end
    return nothing
end

# Update sincrono de Lenia (FFT real + planes/buffers cacheados). La version original esta
# abajo como lenia_model_step_legacy!.
function lenia_model_step!(model)
    props = abmproperties(model)
    haskey(props, :_lenia_state) || _lenia_build_kernel!(model)   # init perezoso si no hubo post_init
    st  = props[:_lenia_state]
    μ   = Float64(props[:lenia_mu])
    σ   = Float64(props[:lenia_sigma])
    dt  = Float64(props[:dt])
    gfn = get(props, :growth_fn, _lenia_growth_fn)
    _lenia_run_step!(model, st, μ, σ, dt, gfn)
end

# Version original (FFT compleja, sin planes ni buffers). Se conserva como referencia.
function lenia_model_step_legacy!(model)
    dims = size(abmspace(model))
    A = zeros(Float64, dims...)
    for agent in allagents(model)
        A[agent.pos[1], agent.pos[2]] = agent.state
    end
    U = real(ifft(fft(A) .* model.kernel_fft))
    μ   = Float64(model.lenia_mu)
    σ   = Float64(model.lenia_sigma)
    dt  = Float64(model.dt)
    gfn = get(abmproperties(model), :growth_fn, _lenia_growth_fn)
    A_new = clamp.(A .+ dt .* gfn.(U, μ, σ), 0.0, 1.0)
    for agent in allagents(model)
        agent.state = A_new[agent.pos[1], agent.pos[2]]
    end
end

# Lenia con ruido gaussiano opcional dentro de una ventana temporal. Sin sigma_noise (o 0)
# se comporta como lenia_model_step!. El ruido usa abmrng(model), asi que es reproducible.
# Parametros en abmproperties(model):
#   sigma_noise       sigma del ruido (0 = sin ruido)
#   noise_start_step  primer paso con ruido
#   noise_end_step    ultimo paso con ruido
#   noise_scope       "support" (solo A>eps) | "field" (todo el campo)
function lenia_noisy_step!(model)
    lenia_model_step!(model)                     # update Lenia estándar (clamp a [0,1])

    props = abmproperties(model)
    props[:step_count] = get(props, :step_count, 0) + 1
    t = props[:step_count]

    σ = Float64(get(props, :sigma_noise, 0.0))
    σ <= 0.0 && return                           # sin ruido: Lenia determinista

    k0 = Int(get(props, :noise_start_step, typemax(Int)))
    k1 = Int(get(props, :noise_end_step,   typemax(Int)))
    (t < k0 || t > k1) && return                 # fuera de la ventana de perturbación

    scope = string(get(props, :noise_scope, "support"))
    eps   = 1e-3
    rng   = abmrng(model)
    for agent in allagents(model)
        (scope == "support" && agent.state <= eps) && continue   # perturbar solo el organismo
        agent.state = clamp(agent.state + σ * randn(rng), 0.0, 1.0)
    end
end

# --- Metricas (se nombran en [run].mdata) ---
total_energy(model) = sum(agent.state for agent in allagents(model))           # energia total
mean_state(model)   = total_energy(model) / max(1, nagents(model))             # densidad media
active_cells(model) = count(agent -> agent.state > 1e-3, allagents(model))     # celdas activas

# --- PREDATOR PREY (stub) ---
function predator_prey_step!(_agent, _model)
end


end