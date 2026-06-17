# Exporta la semilla orbium (20x20) desde el modulo CustomEvolutionRules a un fichero de texto
# compartido por TODAS las implementaciones del benchmark (C, Python, Julia, GPU).
# Fuente unica de verdad => semilla identica en todos los lenguajes, sin riesgo de transcripcion.
#
# Formato de bench/orbium_seed.txt:
#   linea 1:  "<ph> <pw>"          (dimensiones del patron, p.ej. 20 20)
#   lineas siguientes: ph filas de pw valores Float64 separados por espacio (row-major)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CustomEvolutionRules.jl"))
using .CustomEvolutionRules

const ORB = CustomEvolutionRules.ORBIUM   # Matrix{Float64}, ORBIUM[i,j] = patron fila i, col j
ph, pw = size(ORB)

open(joinpath(@__DIR__, "orbium_seed.txt"), "w") do io
    println(io, "$ph $pw")
    for i in 1:ph
        println(io, join((repr(ORB[i, j]) for j in 1:pw), " "))
    end
end

println("orbium_seed.txt escrito: $(ph)x$(pw), suma=", sum(ORB), " max=", maximum(ORB))
