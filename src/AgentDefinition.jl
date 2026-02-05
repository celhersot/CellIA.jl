module AgentDefinition

export GoLAgent

using Agents

@agent struct GoLAgent(GridAgent{2})
    status::Bool # true = vivo, false = muerto
end

end