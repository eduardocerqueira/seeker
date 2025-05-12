#date: 2025-05-12T17:11:02Z
#url: https://api.github.com/gists/7890bf3ec2c96f075e34f3d9ee726f73
#owner: https://api.github.com/users/augustoaa1

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Parámetros físicos
mass1_source = 30  # masas en M_sun (no en SI)
mass2_source = 20
redshift = 0.09

# Redshifteamos las masas observadas
mass1 = mass1_source * (1 + redshift)
mass2 = mass2_source * (1 + redshift)

distance = 410  # en Mpc

# Parámetros de la señal
f_lower = 20
f_upper = 300
sample_rate = 4096
duration = 4  # en segundos

# Constantes físicas
G = 6.67430e-11  # Constante gravitacional en m^3 kg^-1 s^-2
c = 2.99792458e8  # Velocidad de la luz en m/s
Msun = 1.989e30   # Masa solar en kg
Mpc = 3.086e22    # Megaparsec en metros

def generate_waveform(mass1, mass2, distance, f_lower, f_upper, sample_rate, duration):
    """
    Genera una forma de onda gravitacional similar a IMRPhenomD para un sistema binario
    """
    # Convertir unidades
    m1_kg = mass1 * Msun
    m2_kg = mass2 * Msun
    dist_m = distance * Mpc
    
    # Calcular parámetros derivados
    total_mass = mass1 + mass2
    eta = (mass1 * mass2) / (total_mass ** 2)  # Razón de masas simétrica
    chirp_mass = total_mass * (eta ** (3/5))  # Masa de chirp
    
    # Tiempo de la señal
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Tiempo de coalescencia (asumimos que ocurre cerca del final)
    t_coal = duration * 0.95
    
    # Fase de inspiral: frecuencia aumenta según una aproximación de la ecuación de chirp
    # La frecuencia instantánea sigue aproximadamente f(t) ∝ (tc - t)^(-3/8)
    f_t = np.zeros_like(t)
    
    # Solo calculamos para t < t_coal
    mask = t < t_coal
    t_to_coal = t_coal - t[mask]
    
    # Escalamiento de frecuencia
    f_scale = (chirp_mass / 30) ** (-5/8)
    
    # Frecuencia instantánea como función del tiempo
    f_t[mask] = f_lower + (f_upper - f_lower) * (1 - (t_to_coal / t_coal) ** (3/8)) * f_scale
    
    # Amplitud como función del tiempo y frecuencia - aumenta a medida que se acerca la coalescencia
    # h ∝ (chirp_mass)^(5/3) * f^(2/3) / distance
    amp_scale = 1e-21 * ((chirp_mass * Msun) ** (5/3)) / dist_m
    amplitude = np.zeros_like(t)
    amplitude[mask] = amp_scale * (np.pi * f_t[mask]) ** (2/3)
    
    # Fase (integración numérica aproximada de la frecuencia)
    phase = np.zeros_like(t)
    phase[mask] = 2 * np.pi * np.cumsum(f_t[mask]) * (t[1] - t[0])
    
    # Forma de onda h+
    h_plus = amplitude * np.cos(phase)
    
    # Después de la coalescencia (t > t_coal), añadimos una señal de ringdown que decae exponencialmente
    if np.any(~mask):
        # Frecuencia del ringdown (depende de la masa final)
        f_ring = f_upper * 1.2
        
        # Tiempo de decaimiento
        tau = 4 * (total_mass / 100) * (G * total_mass * Msun) / (c ** 3)
        
        # Amplitud inicial del ringdown (continuidad con la última amplitud del inspiral)
        amp_ring_init = amplitude[mask][-1]
        
        # Ringdown: oscilación amortiguada
        t_ring = t[~mask] - t_coal
        h_ring = amp_ring_init * np.exp(-t_ring / tau) * np.cos(2 * np.pi * f_ring * t_ring + phase[mask][-1])
        
        h_plus[~mask] = h_ring
    
    return h_plus

def generate_noise(length, delta_t, psd_model='aLIGO'):
    """
    Genera ruido gaussiano coloreado según un modelo de PSD
    
    :param length: Longitud de la serie temporal
    :param delta_t: Intervalo de tiempo entre muestras
    :param psd_model: Modelo de PSD a usar ('aLIGO' por defecto)
    :return: Serie temporal de ruido
    """
    # Frecuencias para la PSD
    freqs = np.fft.rfftfreq(length, delta_t)
    
    # Evitamos f=0
    freqs[0] = freqs[1] / 2
    
    # Implementación simplificada de PSD aLIGO
    if psd_model == 'aLIGO':
        # Parámetros aproximados para aLIGO Zero Det. High Power
        fs = 20.0  # Frecuencia de rodilla baja
        a = 1e-47  # Factor de escala
        b = 16.0   # Exponente para el término de baja frecuencia
        c = 2.0    # Exponente para el término de alta frecuencia
        
        # Modelo simplificado: S(f) = a * (1 + (f/fs)^-b + (f/fs)^c)
        psd = a * (1 + (freqs/fs)**(-b) + (freqs/fs)**c)
    else:
        # PSD plana por defecto
        psd = np.ones_like(freqs)
    
    # Generar ruido blanco en el dominio de la frecuencia
    nf = np.sqrt(psd/2) * (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs)))
    nf[0] = nf[0].real  # Componente DC debe ser real
    
    # Transformada inversa para obtener ruido en el dominio del tiempo
    noise = np.fft.irfft(nf) * np.sqrt(length)
    
    return noise

# Generar la onda gravitacional
hp = generate_waveform(mass1, mass2, distance, f_lower, f_upper, sample_rate, duration)

# Redimensionar la señal para que coincida con la duración deseada
target_length = int(sample_rate * duration)

# Generar el ruido (simulando la PSD de aLIGO)
noise_hanford = generate_noise(target_length, 1.0/sample_rate, psd_model='aLIGO')
noise_livingston = generate_noise(target_length, 1.0/sample_rate, psd_model='aLIGO')

# Ajustar la amplitud del ruido para que sea similar a la señal (SNR realista)
signal_rms = np.sqrt(np.mean(hp**2))
noise_rms = np.sqrt(np.mean(noise_hanford**2))
snr_target = 10  # Relación señal-ruido objetivo
noise_scale_factor = signal_rms / (noise_rms * snr_target)

noise_hanford *= noise_scale_factor
noise_livingston *= noise_scale_factor

# Señal con ruido
signal_hanford = hp + noise_hanford
signal_livingston = hp + noise_livingston

# Ejes de tiempo
time = np.arange(target_length) / sample_rate

# Mostrar una tabla con algunas filas
print("Time (s)     | Hanford Strain   | Livingston Strain")
print("-----------------------------------------------------")
for t, h1, h2 in zip(time[:10], signal_hanford[:10], signal_livingston[:10]):
    print(f"{t:.6f}   | {h1:.6e}     | {h2:.6e}")

# Graficar las señales en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.plot(time, signal_hanford, label='Hanford', alpha=0.8)
plt.plot(time, signal_livingston, label='Livingston', alpha=0.6)
plt.xlabel("Tiempo (s)")
plt.ylabel("Strain")
plt.title("Señales simuladas con ruido - Hanford y Livingston")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Espectrograma de la señal de Hanford
plt.figure(figsize=(10, 5))
plt.specgram(signal_hanford, NFFT=256, Fs=sample_rate, noverlap=128, cmap='viridis')
plt.title("Espectrograma de la señal simulada (Hanford)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.colorbar(label="Intensidad (dB)")
plt.tight_layout()
plt.show()