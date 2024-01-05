#date: 2024-01-05T17:10:24Z
#url: https://api.github.com/gists/5dadb7e22dbb06102284adb88c721ae3
#owner: https://api.github.com/users/terjehaukaas

# ------------------------------------------------------------------------
# The following Python code is implemented by Professor Terje Haukaas at
# the University of British Columbia in Vancouver, Canada. It is made
# freely available online at terje.civil.ubc.ca together with notes,
# examples, and additional Python code. Please be cautious when using
# this code; it may contain bugs and comes without warranty of any kind.
# ------------------------------------------------------------------------

from G2AnalysisNonlinearStatic import *
from G2Model import *

#              |
#              | P
#              |
#              V
#        ----> * -----> F
#        ----> |
#        ----> |
#        ----> |
#        ----> |
#      q ----> | L
#        ----> |
#        ----> |
#        ----> |
#        ----> |
#        ----> |
#            -----

# Input [N, m, kg, sec]
elementType = 12            # 12 = Displacement-based frame element / 13 = Force-based frame element
materialType = 'BoucWen'   # 'Bilinear' / 'Plasticity' / 'BoucWen'
L = 5.0                     # Total length of cantilever
nel = 5                     # Number of elements along cantilever
F = 0                       # Point load
P = 0.0                     # Axial force
q = 100e3                   # Distributed load
E = 200e9                   # Modulus of elasticity
fy = 350e6                  # Yield stress
alpha = 0.02                # Second-slope stiffness
eta = 3                     # Bouc-Wen sharpness
gamma = 0.5                 # Bouc-Wen parameter
beta = 0.5                  # Bouc-Wen parameter
H = alpha*E/(1-alpha)       # Kinematic hardening parameter
K = 0                       # Linear isotropic hardening parameter
delta = 0                   # Saturation isotropic hardening parameter
fy_inf = 100.0              # Asymptotic yield stress for saturation isotropic hardening
hw = 0.355                  # Web height
bf = 0.365                  # Flange width
tf = 0.018                  # Flange thickness
tw = 0.011                  # Web thickness
nf = 3                      # Number of fibers in the flange
nw = 8                      # Number of fibres in the web
nsec = 5                    # Number of integration points
nsteps = 12                 # Number of pseudo-time steps, each of length dt
dt = 0.1                    # Delta-t
KcalcFrequency = 1          # 0=initial stress method, 1=Newton-Raphson, maxIter=Modified NR
maxIter = 100               # Maximum number of equilibrium iterations in the Newton-Raphson algorithm
tol = 1e-5                  # Convergence tolerance for the Newton-Raphson algorithm
trackNode = nel+1           # Node to be plotted
trackDOF = 1                # DOF to be plotted

# Nodal coordinates
NODES = []
for i in range(nel+1):
    NODES.append([0.0, i*L/nel])

# Boundary conditions (0=free, 1=fixed, sets #DOFs per node)
CONSTRAINTS = [[1, 1, 1]]
for i in range(nel):
    CONSTRAINTS.append([0, 0, 0])

# Element information
ELEMENTS = []
for i in range(nel):
    ELEMENTS.append([elementType, nsec, q, i+1, i+2])

# Section information (one section per element)
SECTIONS = []
for i in range(nel):
    SECTIONS.append(['WideFlange', hw, bf, tf, tw, nf, nw])

# Material information (one material per element)
MATERIALS = []
for i in range(nel):
    if materialType == 'Bilinear':
        MATERIALS.append(['Bilinear', E, fy, alpha])
    elif materialType == 'Plasticity':
        MATERIALS.append(['Plasticity', E, fy, H, K, delta, fy_inf])
    elif materialType == 'BoucWen':
        MATERIALS.append(['BoucWen', E, fy, alpha, eta, beta, gamma])
    else:
        print('\n'"Error: Wrong material type")
        import sys
        sys.exit()

# Nodal loads
LOADS = np.zeros((nel+1, 3))
LOADS[nel, 0] = F
LOADS[nel, 1] = -P

# Lumped mass
MASS = [[0, 0, 0]]
for i in range(nel):
    MASS.append([0, 0, 0])

# Create the model object
a = [NODES, CONSTRAINTS, ELEMENTS, SECTIONS, MATERIALS, LOADS, MASS]
m = model(a)

# Run the analysis and plot the response
loadFactor, response, ddmSensitivity = nonlinearStaticAnalysis(m, nsteps, dt, maxIter, KcalcFrequency, tol, trackNode, trackDOF, [])
