using Agents
using Random, LinearAlgebra

@agent struct Bird(ContinuousAgent{2, Float64})
    const speed::Float64
    const cohere_factor::Float64
    const separation::Float64
    const separate_factor::Float64
    const match_factor::Float64
    const visual_distance::Float64
end

function agent_step!(bird, model)
    # Obtain the ids of neighbors within the bird's visual distance
    neighbor_agents = nearby_agents(bird, model, bird.visual_distance)
    N = 0
    match = separate = cohere = SVector{2}(0.0, 0.0)
    # Calculate behaviour properties based on neighbors
    for neighbor in neighbor_agents
        N += 1
        heading = get_direction(bird.pos, neighbor.pos, model)

        # `cohere` computes the average position of neighboring birds
        cohere += heading
        # `match` computes the average trajectory of neighboring birds
        match += neighbor.vel
        if sum(heading .^ 2) < bird.separation^2
            # `separate` repels the bird away from neighboring birds
            separate -= heading
        end
    end

    # Normalise results based on model input and neighbor count
    cohere *= bird.cohere_factor
    separate *= bird.separate_factor
    match *= bird.match_factor
    # Compute velocity based on rules defined above
    bird.vel += (cohere + separate + match) / max(N, 1)
    bird.vel /= norm(bird.vel)
    # Move bird according to new velocity and speed
    return move_agent!(bird, model, bird.speed)
end