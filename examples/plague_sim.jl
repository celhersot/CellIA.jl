using Agents
using Random

struct SickState
    status::Symbol  # :sick, :healthy
    recovery_time::Int
end

function SickState(s::String)
    parts = split(replace(s, r"SickState\(|\)" => ""), ",") # It performs the necessary processing with the characters, simplified here as an example.
    status = Symbol(strip(parts[1]))
    recovery_time = parse(Int, strip(parts[2]))
    return SickState(status, recovery_time)
end

function plague_step!(agent, model)
    current = agent.state
    
    if current.status == :sick
        if rand() < model.death_rate
            model.next_states[agent.id] = SickState(:healthy, 0)
        else
            recovery_time = max(agent.recovery_time - 1, 0)
            if recovery_time == 0
                model.next_states[agent.id] = SickState(:healthy, 0)
            else
                model.next_states[agent.id] = SickState(:sick, recovery_time)
            end
        end
    end
end