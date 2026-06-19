using Test
using Agents

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "spaces", "HexagonalSpace.jl"))
include(joinpath(SRC, "UniversalAgents.jl"))
include(joinpath(SRC, "CustomEvolutionRules.jl"))
include(joinpath(SRC, "SpaceDefinition.jl"))
include(joinpath(SRC, "Initialization.jl"))
using .SpaceDefinition
using .Initialization

# Campo de estados por posición (para comparar modelos sobre grid).
function field(m)
    d = size(abmspace(m))
    A = zeros(Float64, d...)
    for a in allagents(m)
        A[a.pos[1], a.pos[2]] = Float64(a.state)
    end
    return A
end

@testset "Cell_IA" begin

    @testset "create_space" begin
        grid = create_space(Dict("space" => Dict("type" => "grid", "dimensions" => [10, 10])))
        @test size(grid) == (10, 10)
        cont = create_space(Dict("space" => Dict("type" => "continuous", "dimensions" => [10, 10]),
                                 "agents" => Dict()))
        @test cont isa ContinuousSpace
        hex = create_space(Dict("space" => Dict("type" => "hexagonal", "dimensions" => [8, 8])))
        @test size(hex) == (8, 8)
    end

    @testset "Game of Life (Bool)" begin
        cfg = Dict(
            "simulation" => Dict("seed" => 42),
            "space"      => Dict("type" => "grid", "dimensions" => [20, 20], "periodic" => true),
            "agents"     => Dict("state_type" => "Bool"),
            "population" => Dict("pop_density" => Dict("true" => 0.3, "false" => 0.7)),
            "properties" => Dict("min_to_live" => 2, "max_to_live" => 3),
            "rules"      => Dict("agent_step" => "gol_step!", "model_step" => "default_model_step!",
                                 "initialization_rule" => "random"),
        )
        m = initialize_model(cfg)
        @test nagents(m) == 400
        step!(m, 5)
        @test nagents(m) == 400
        @test all(a -> a.state isa Bool, allagents(m))
    end

    @testset "Rock-Paper-Scissors (Symbol)" begin
        cfg = Dict(
            "simulation" => Dict("seed" => 1),
            "space"      => Dict("type" => "grid", "dimensions" => [30, 30], "periodic" => true),
            "agents"     => Dict("state_type" => "Symbol"),
            "population" => Dict("pop_density" => Dict("rock" => 0.34, "paper" => 0.33, "scissors" => 0.33)),
            "properties" => Dict("threshold" => 3),
            "rules"      => Dict("agent_step" => "rps_step!", "model_step" => "default_model_step!",
                                 "initialization_rule" => "random"),
        )
        m = initialize_model(cfg)
        step!(m, 5)
        @test all(a -> a.state in (:rock, :paper, :scissors), allagents(m))
    end

    @testset "Lenia: optimizado == legacy" begin
        lenia_cfg(stepfn) = Dict(
            "simulation" => Dict("seed" => 42),
            "space"      => Dict("type" => "grid", "dimensions" => [32, 32], "periodic" => true,
                                 "metric" => "chebyshev"),
            "agents"     => Dict("state_type" => "Float64"),
            "population" => Dict("pop_density" => Dict("0.0" => 1.0)),
            "properties" => Dict("lenia_mu" => 0.15, "lenia_sigma" => 0.015, "dt" => 0.1,
                                 "kernel_radius" => 13, "kernel_type" => "gaussian"),
            "rules"      => Dict("initialization_rule" => "uniform_float", "post_init" => "lenia_init!",
                                 "model_step" => stepfn),
        )
        m_opt = initialize_model(lenia_cfg("lenia_model_step!"))
        m_leg = initialize_model(lenia_cfg("lenia_model_step_legacy!"))
        @test field(m_opt) == field(m_leg)          # mismo campo inicial (mismo seed)
        step!(m_opt, 5)
        step!(m_leg, 5)
        @test maximum(abs.(field(m_opt) .- field(m_leg))) < 1e-6

        @test CustomEvolutionRules.total_energy(m_opt) ≈ sum(field(m_opt))
        @test CustomEvolutionRules.active_cells(m_opt) == count(>(1e-3), field(m_opt))
    end

    @testset "image trace: 1 agente/celda y asentamiento" begin
        Base.include(Main.CustomEvolutionRules,
                     abspath(joinpath(@__DIR__, "..", "examples", "image_trace.jl")))
        cfg = Dict(
            "simulation" => Dict("seed" => 42),
            "space"      => Dict("type" => "grid", "dimensions" => [40, 40], "periodic" => false,
                                 "metric" => "chebyshev"),
            "agents"     => Dict("state_type" => "Painter"),
            "properties" => Dict("image_path" => "", "density" => 0.7, "search_radius" => 6,
                                 "chameleon" => true, "chameleon_rate" => 0.2,
                                 "arrival_tol" => 0.12, "patience" => 3),
            "rules"      => Dict("initialization_rule" => "empty", "post_init" => "trace_init!",
                                 "model_step" => "trace_model_step!"),
        )
        m = Base.invokelatest(initialize_model, cfg)
        @test nagents(m) > 0
        for _ in 1:30
            Base.invokelatest(step!, m, 1)
            pos = [a.pos for a in allagents(m)]
            @test length(pos) == length(unique(pos))   # nunca dos agentes en la misma celda
        end
        ff = getfield(Main.CustomEvolutionRules, :found_fraction)
        @test Base.invokelatest(ff, m) > 0.8
    end
end
