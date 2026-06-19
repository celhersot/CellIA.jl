module SpaceDefinition
using Agents
using ..HexagonalSpace
export create_space

"""
    create_space(config::Dict)

Crea el espacio de la simulación a partir de `config["space"]`. Según el campo `type`
devuelve un `GridSpaceSingle` (`"grid"`), un `ContinuousSpace` (`"continuous"`) o un
`HexagonalGridSpace` (`"hexagonal"`), usando `dimensions`, `periodic` y `metric`.
Es el primer paso de `initialize_model`.
"""
function create_space(config::Dict)
    space_conf = config["space"]
    type = space_conf["type"]
    dims = Tuple(space_conf["dimensions"])
    periodic = get(space_conf, "periodic", false)
    metric = Symbol(get(space_conf, "metric", "chebyshev"))

    if type == "grid"
        return GridSpaceSingle(dims; periodic = periodic, metric = metric)
    elseif type == "hexagonal"
        return HexagonalGridSpace(dims; periodic = periodic)
    elseif type == "continuous"
        extent = Float64.(dims)
        # spacing is optional: explicit [space].spacing, else derived from
        # [agents].visual_distance if present, else a sensible default.
        spacing = if haskey(space_conf, "spacing")
            Float64(space_conf["spacing"])
        elseif haskey(get(config, "agents", Dict()), "visual_distance")
            Float64(config["agents"]["visual_distance"]) / 1.5
        else
            minimum(extent) / 10.0
        end
        return ContinuousSpace(extent; periodic = periodic, spacing = spacing)
    else
        error("Unrecognized or unsupported space type: '$type'")
    end
end

end