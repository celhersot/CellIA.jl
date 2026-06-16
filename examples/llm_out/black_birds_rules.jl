using Agents
using Random, LinearAlgebra

@agent struct BlackBird(ContinuousAgent{2, Float64})
    const speed::Float64
    const cohere_factor::Float64
    const separation::Float64
    const separate_factor::Float64
    const match_factor::Float64
    const visual_distance::Float64
end

function agent_step!(bird, model)
    neighbor_agents = nearby_agents(bird, model, bird.visual_distance)
    N = 0
    match = separate = cohere = SVector{2}(0.0, 0.0)
    for neighbor in neighbor_agents
        N += 1
        heading = get_direction(bird.pos, neighbor.pos, model)
        cohere += heading
        match  += neighbor.vel
        if sum(heading .^ 2) < bird.separation^2
            separate -= heading
        end
    end
    cohere   *= bird.cohere_factor
    separate *= bird.separate_factor
    match    *= bird.match_factor
    bird.vel += (cohere + separate + match) / max(N, 1)
    bird.vel /= norm(bird.vel)
    return move_agent!(bird, model, bird.speed)
end
