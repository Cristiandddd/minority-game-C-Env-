import numpy as np
import pandas as pd

# Parámetros dados
M = 9
log_alpha_min = np.log10(0.01)
log_alpha_max = np.log10(100)
num_points = 50

# Generar valores logarítmicamente espaciados
log_alpha_values = np.linspace(log_alpha_min, log_alpha_max, num_points)
alpha_values = 10**log_alpha_values

# Calcular valores de N (alpha = 2^M / N)
N_values = (2**M) / alpha_values

# Crear DataFrame
df = pd.DataFrame({
    'alpha': alpha_values,
    'N': N_values
})

# Exportar a CSV
df.to_csv('alpha_N_values.csv', index=False, float_format='%.6f')
