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

# --- LENIA (Continuous Cellular Automata via FFT convolution) ---
#
# Default kernel: smooth bump (Gaussian-like) used in canonical Lenia.
# Default growth: bell curve centered at μ, width σ.
# Both can be overridden per-simulation by storing functions in abmproperties(model):
#   abmproperties(model)[:kernel_fn] = r -> <your kernel>    (r ∈ [0,1])
#   abmproperties(model)[:growth_fn] = (u,μ,σ) -> <your growth>

_lenia_kernel_fn(r::Float64) = r < 1.0 ? exp(4.0 - 1.0 / (r * (1.0 - r) + 1e-10)) : 0.0

_lenia_growth_fn(u::Float64, μ::Float64, σ::Float64) =
    2.0 * exp(-((u - μ)^2) / (2.0 * σ^2)) - 1.0

# _lenia_build_kernel! — precomputes the spatial kernel and stores its FFT.
# Called from lenia_init!; also callable from user-defined post_init functions
# after setting abmproperties(model)[:kernel_fn].
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
    abmproperties(model)[:kernel_fft] = fft(K)
end

# lenia_init! — default post_init hook.
# Builds the kernel FFT from abmproperties(model)[:kernel_fn] (or the default).
# State randomisation is handled separately via initialization_rule = "uniform_float".
function lenia_init!(model)
    kernel_type = string(get(abmproperties(model), :kernel_type, "gaussian"))
    if kernel_type == "polynomial"
        abmproperties(model)[:kernel_fn] = (r::Float64) -> max(0.0, 1.0 - r^2)^4
    end
    # kernel_type == "custom" → user already set abmproperties(model)[:kernel_fn] earlier
    _lenia_build_kernel!(model)
end

# lenia_model_step! — full synchronous update via FFT convolution.
# Does NOT use next_states; reads A first, writes back after, so update is coherent.
function lenia_model_step!(model)
    dims = size(abmspace(model))

    # 1. Snapshot current state into a matrix
    A = zeros(Float64, dims...)
    for agent in allagents(model)
        A[agent.pos[1], agent.pos[2]] = agent.state
    end

    # 2. U = K ⊛ A  (circular convolution via FFT)
    U = real(ifft(fft(A) .* model.kernel_fft))

    # 3. Apply growth function element-wise and clamp
    μ   = Float64(model.lenia_mu)
    σ   = Float64(model.lenia_sigma)
    dt  = Float64(model.dt)
    gfn = get(abmproperties(model), :growth_fn, _lenia_growth_fn)

    A_new = clamp.(A .+ dt .* gfn.(U, μ, σ), 0.0, 1.0)

    # 4. Write new states back to agents
    for agent in allagents(model)
        agent.state = A_new[agent.pos[1], agent.pos[2]]
    end
end

# --- PREDATOR PREY (stub) ---
function predator_prey_step!(_agent, _model)
end


end