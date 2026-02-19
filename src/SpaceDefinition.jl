module SpaceDefinition
using Agents
export create_space

function create_space(config::Dict)
    space_conf = config["space"]
    type = space_conf["type"]
    dims = Tuple(space_conf["dimensions"])
    periodic = get(space_conf, "periodic", true)
    metric = Symbol(get(space_conf, "metric", "chebyshev"))

    if type == "grid"
        return GridSpaceSingle(dims; periodic  =periodic, metric = metric)
    elseif type == "hexagonal"
        # Future implementation
        error("Hexagonal space is a future implementation.")
    elseif type == "continuous"
        error("Continuous space is a future implementation.")
    else
        error("Unrecognized or unsupported space type: '$type'")
    end
end

end