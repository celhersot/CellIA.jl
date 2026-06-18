# Reglas del experimento de resiliencia de Lenia: solo lo propio de este caso (la semilla
# del orbium). El paso con ruido (lenia_noisy_step!) y las metricas son built-ins genericos
# de CustomEvolutionRules.jl, donde main.jl inyecta este archivo.

# Orbium (Bert Chan): planeador estable con los parametros por defecto (mu=0.15, sigma=0.015,
# R=13, dt=0.1, kernel gaussiano). Da una estructura estable que perturbar. Celdas en [0,1], 20x20.
const _ORBIUM = [
0.0  0.0  0.0  0.0  0.0  0.0  0.10 0.14 0.10 0.0  0.0  0.03 0.03 0.0  0.0  0.30 0.0  0.0  0.0  0.0 ;
0.0  0.0  0.0  0.0  0.0  0.08 0.24 0.30 0.30 0.18 0.14 0.15 0.16 0.15 0.09 0.20 0.0  0.0  0.0  0.0 ;
0.0  0.0  0.0  0.0  0.0  0.15 0.34 0.44 0.46 0.38 0.18 0.14 0.11 0.13 0.19 0.18 0.45 0.0  0.0  0.0 ;
0.0  0.0  0.0  0.0  0.06 0.13 0.39 0.50 0.50 0.37 0.06 0.0  0.0  0.0  0.02 0.16 0.68 0.0  0.0  0.0 ;
0.0  0.0  0.0  0.11 0.17 0.17 0.33 0.40 0.38 0.28 0.14 0.0  0.0  0.0  0.0  0.0  0.18 0.42 0.0  0.0 ;
0.0  0.0  0.09 0.18 0.13 0.06 0.08 0.26 0.32 0.32 0.27 0.0  0.0  0.0  0.0  0.0  0.0  0.82 0.0  0.0 ;
0.27 0.0  0.16 0.12 0.0  0.0  0.0  0.25 0.38 0.44 0.45 0.34 0.0  0.0  0.0  0.0  0.0  0.22 0.17 0.0 ;
0.0  0.07 0.20 0.02 0.0  0.0  0.0  0.31 0.48 0.57 0.60 0.57 0.0  0.0  0.0  0.0  0.0  0.0  0.49 0.0 ;
0.0  0.59 0.19 0.0  0.0  0.0  0.0  0.20 0.57 0.69 0.76 0.76 0.49 0.0  0.0  0.0  0.0  0.0  0.36 0.0 ;
0.0  0.58 0.19 0.0  0.0  0.0  0.0  0.0  0.67 0.83 0.90 0.92 0.87 0.12 0.0  0.0  0.0  0.0  0.22 0.07 ;
0.0  0.0  0.46 0.0  0.0  0.0  0.0  0.0  0.70 0.93 1.0  1.0  1.0  0.61 0.0  0.0  0.0  0.0  0.18 0.11 ;
0.0  0.0  0.82 0.0  0.0  0.0  0.0  0.0  0.47 1.0  1.0  0.98 1.0  0.96 0.27 0.0  0.0  0.0  0.19 0.10 ;
0.0  0.0  0.46 0.0  0.0  0.0  0.0  0.0  0.25 1.0  1.0  0.84 0.92 0.97 0.54 0.14 0.04 0.10 0.21 0.05 ;
0.0  0.0  0.0  0.40 0.0  0.0  0.0  0.0  0.09 0.80 1.0  0.82 0.80 0.85 0.63 0.31 0.18 0.19 0.20 0.01 ;
0.0  0.0  0.0  0.36 0.10 0.0  0.0  0.0  0.05 0.54 0.86 0.79 0.74 0.72 0.60 0.39 0.28 0.24 0.13 0.0 ;
0.0  0.0  0.0  0.01 0.30 0.07 0.0  0.0  0.08 0.36 0.64 0.70 0.64 0.60 0.51 0.39 0.29 0.19 0.04 0.0 ;
0.0  0.0  0.0  0.0  0.10 0.24 0.14 0.10 0.15 0.29 0.45 0.53 0.52 0.46 0.40 0.31 0.21 0.08 0.0  0.0 ;
0.0  0.0  0.0  0.0  0.0  0.08 0.21 0.21 0.22 0.29 0.36 0.39 0.37 0.33 0.26 0.18 0.09 0.0  0.0  0.0 ;
0.0  0.0  0.0  0.0  0.0  0.0  0.03 0.13 0.19 0.22 0.24 0.24 0.23 0.18 0.13 0.05 0.0  0.0  0.0  0.0 ;
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.02 0.06 0.08 0.09 0.07 0.05 0.01 0.0  0.0  0.0  0.0  0.0 ]

# Estampa el orbium centrado en la malla; deja el resto del campo a 0.
function _seed_orbium!(model)
    dims    = size(abmspace(model))
    oh, ow  = size(_ORBIUM)            # filas, columnas del patrón
    off_x   = (dims[1] - ow) ÷ 2
    off_y   = (dims[2] - oh) ÷ 2
    for a in allagents(model)
        x, y = a.pos[1], a.pos[2]
        c = x - off_x
        r = y - off_y
        a.state = (1 <= r <= oh && 1 <= c <= ow) ? _ORBIUM[r, c] : 0.0
    end
end

# post_init del experimento: construye el kernel FFT (built-in lenia_init!) y siembra el orbium.
# Se referencia desde el TOML como rules.post_init = "lenia_perturbation_init!".
function lenia_perturbation_init!(model)
    lenia_init!(model)                          # built-in: construye model.kernel_fft
    _seed_orbium!(model)                        # específico: estampa el orbium
    abmproperties(model)[:step_count] = 0
end
