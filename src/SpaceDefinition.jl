module SpaceDefinition
using Agents
export create_space

function create_space(config::Dict)
    space_conf = config["space"]
    type = space_conf["type"]
    dims = Tuple(space_conf["dimensions"])
    periodic = get(space_conf, "periodic", true)
    metric = Symbol(get(space_conf, "metric", "chebyshev"))
    periodic = space_conf["periodic"]

    #update_vel = space_conf["update_vel"]

    if type == "grid"
        return GridSpaceSingle(dims; periodic = periodic, metric = metric)
    elseif type == "hexagonal"
        # Future implementation
        error("Hexagonal space is a future implementation.")
        return HexagonalSpaceSingle(dims; periodic = false, metric = :chebyshev)
    elseif type == "continuous"
        visual_distance = config["agents"]["visual_distance"]
        extent = Float64.(dims)

        #return ContinuousSpace(extent; periodic = periodic, spacing = spacing, update_vel! = update_vel)
        return ContinuousSpace(extent; periodic = periodic, spacing = visual_distance / 1.5)
    else
        error("Unrecognized or unsupported space type: '$type'")
    end
end

end