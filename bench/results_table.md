# Resultados del benchmark Lenia (mediana ms/step, 1 hilo CPU salvo GPU)

Pasos por simulacion: 1000. Semilla: orbiums teselados (densidad constante, R=13).

| Implementacion | Dispositivo | 128x128 ms/step | 128x128 total (s) | 256x256 ms/step | 256x256 total (s) | 512x512 ms/step | 512x512 total (s) |
|---|---|---|---|---|---|---|---|
| C (FFTW) | CPU | 0.626 | 0.63 | 3.234 | 3.23 | 14.309 | 14.31 |
| Julia puro (FFTW.jl) | CPU | 0.702 | 0.70 | 3.411 | 3.41 | 14.481 | 14.48 |
| Cell_IA optimizado (Agents.jl + FFTW) | CPU | 1.854 | 1.85 | 7.968 | 7.97 | 51.168 | 51.17 |
| Cell_IA legacy (sin optimizar) | CPU | 5.842 | 5.84 | 30.633 | 30.63 | 147.659 | 147.66 |
| Python NumPy (pocketfft) | CPU | 1.753 | 1.75 | 8.204 | 8.20 | 39.270 | 39.27 |
| Python pyFFTW (FFTW) | CPU | 0.885 | 0.88 | 5.663 | 5.66 | 25.933 | 25.93 |
| Python CuPy (cuFFT) [GPU] | GPU | 0.868 | 0.87 | 1.687 | 1.69 | 7.097 | 7.10 |

## Energia final E (verificacion; debe coincidir entre backends FFTW; ligera deriva esperable en pocketfft/cuFFT a horizonte largo)

| Implementacion | E @ 128x128 | E @ 256x256 | E @ 512x512 |
|---|---|---|---|
| C (FFTW) | 303.896910 | 1215.587639 | 4862.350556 |
| Julia puro (FFTW.jl) | 303.896910 | 1215.587639 | 4862.350555 |
| Cell_IA optimizado (Agents.jl + FFTW) | 303.896910 | 1215.587639 | 4862.350556 |
| Cell_IA legacy (sin optimizar) | 303.896910 | 1215.587639 | 4862.350556 |
| Python NumPy (pocketfft) | 303.896910 | 1215.587639 | 4862.350555 |
| Python pyFFTW (FFTW) | 303.896910 | 1215.587639 | 4862.350556 |
| Python CuPy (cuFFT) [GPU] | 303.896910 | 1215.587639 | 4862.350554 |
