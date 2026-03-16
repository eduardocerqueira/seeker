#date: 2026-03-16T17:38:32Z
#url: https://api.github.com/gists/7550592da1ad766146e87e6be65cb837
#owner: https://api.github.com/users/andresjros

import math

# Parámetros de la NTC
R0 = 10000  # Ohmios
T0 = 25 + 273.15  # Kelvin
B = 3950  # Constante Beta

# Tolerancia en porcentaje (ej. 1% -> 0.01)
tolerancia = 0.01

def calcular_temperatura(R):
    T = 1 / (1/T0 + (1/B) * math.log(R / R0))  # Kelvin
    return T - 273.15  # Celsius

def estimar_incertidumbre(R):
    delta_R = R * tolerancia
    T1 = calcular_temperatura(R + delta_R)
    T2 = calcular_temperatura(R - delta_R)
    return abs(T1 - T2) / 2

# === CAMBIA AQUÍ EL VALOR DE LA RESISTENCIA ===
R_medida = 8500  # ejemplo: 8500 ohmios

# === EJECUTAR LOS CÁLCULOS ===
T = calcular_temperatura(R_medida)
incertidumbre = estimar_incertidumbre(R_medida)

# Mostrar resultados
print("R = ", R_medida, "ohm")
print("Temperatura = ", round(T, 2), "°C")
print("± ", round(incertidumbre, 2), "°C")
