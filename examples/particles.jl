using Agents
using Random, LinearAlgebra

# A continuous-space agent with DIFFERENT fields than Bird, to verify that
# populate_continuous_world! reads struct fields generically from [agents].
@agent struct Particle(ContinuousAgent{2, Float64})
    const speed::Float64
    const attraction::Float64
    const visual_distance::Float64
end

function agent_step!(p, model)
    neighbors = nearby_agents(p, model, p.visual_distance)
    center = SVector{2}(0.0, 0.0)
    N = 0
    for n in neighbors
        center += get_direction(p.pos, n.pos, model)
        N += 1
    end
    if N > 0
        p.vel += (center / N) * p.attraction
        nrm = norm(p.vel)
        nrm > 0 && (p.vel /= nrm)
    end
    return move_agent!(p, model, p.speed)
end
