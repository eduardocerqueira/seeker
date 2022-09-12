#date: 2022-09-12T17:01:27Z
#url: https://api.github.com/gists/6da738e730b9c8aacf3f32a518472b1d
#owner: https://api.github.com/users/salomaestro

import numpy as np
import matplotlib.pyplot as plt


def vec(x, y):
    """
    En funksjon som lager en 2-dimensjonal vektor (numpy array).

    Argumenter
    ----------
    x, y: int | float
        x og y posisjon til vektoren.
    
    Returnerer
    -------
    numpy.ndarray(x, y)
        Vektoren i 2-dimensjoner.
    """
    return np.array([x, y])

def mag(vec):
    """
    En funksjon som regner ut størrelsen til vektoren du putter inn.

    Argumenter
    ----------
    vec: numpy.ndarray(x, y)
        Vektoren som du vil ha størrelsen til.
    
    Returnerer
    ----------
    float (skalar)
        kvadratrota til summen av kvadratene til hver vektor komponent.
    """
    return np.sqrt(vec[0]**2 + vec[1]**2)

def hat(vec):
    """
    En funksjon som finner retningen i vektorform med størrelse 1 til vektoren du putter inn.

    PS: Dersom størrelsen til vektoren er null ||vector|| = 0 returnerer funksjonen samme vektor sum du putta inn i funksjonen.

    Argumenter
    ----------
    vec: numpy.ndarray(x, y)
        Vektoren funksjonen finner retningen til.
    
    Returnerer
    ----------
    vec: numpy.ndarray(x, y)
        En vektor med samme retning, men størrelse 1 som den som ble puttet inn i funksjonen.
    """

    magnitude = mag(vec)

    if magnitude == 0:
        return vec
    else:
        return vec / magnitude

def wind(y):
    """
    Funksjon for å finne vindhastigheten for en gitt høyde y.

    Argumenter
    ----------
    y: float
        Høyden (skalar) du vil ha vindhastigheten til.
    
    Returnerer
    ----------
    vec: numpy.ndarray(x, y)
        Vektor som varierer avhengig av høyden i input.
    """
    yr = 5
    ur = vec(7, 0)
    return ur * (y / yr) ** (1/7)

# Startbetingelser, setter starthøyde til 4000
pos = vec(0, 4000)
vel = vec(0, 0)

# Konstanter (Går ut fra at vi regner med en ball, derav arealet A)
g = 9.81
m = 70
rho = 1.225
Cd = 0.2
A = np.pi * (0.5) ** 2

# Gravitasjonskraften (Konstant)
Fg = - m * g * vec(0, 1)

# Tidsbetingelser
t = 0
dt = 0.1
tmax = 100

# lister for å large verdier gjennom bevegelsen
time = []
xpos = []
ypos = []

# Ønsker kun å iterere så lenge y-posisjonen er større eller lik 0
while pos[1] >= 0:

    # Regner ut farten til vinden
    u = wind(pos[1])

    # Bruker den relative hastigheten
    rel_vel = vel - u

    # Regner ut kraften vinden påfører ballen
    Fw = - 0.5 * A * rho * Cd * mag(rel_vel) ** 2 * hat(rel_vel)

    # Regner ut kraften lufta påfører ballen
    Fl = - 0.5 * A * Cd * rho * mag(vel) ** 2 * hat(vel)

    # Summen av kreftene (Newtons 2.lov)
    Fnet = Fg + Fl + Fw

    # Euler-Cromer
    vel = vel + (Fnet / m) * dt
    pos = pos + vel * dt

    # oppdaterer tid
    t += dt

    # Lagrer hver enkelt koordinat og tiden.
    xpos.append(pos[0])
    ypos.append(pos[1])
    time.append(t)

# Plotter bevegelsen
plt.plot(xpos, ypos)

# Legger inn at vi ønsker å vise x-aksen fra -1000 til 1000 (ser litt mer realistisk ut)
plt.xlim([-1000, 1000])
plt.show()