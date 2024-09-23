#date: 2024-09-23T17:08:55Z
#url: https://api.github.com/gists/434802b80ced8508706147e331b3709e
#owner: https://api.github.com/users/israelmcmc

import numpy as np
from histpy import Histogram, Axis

def phi_arm_az_phase_space(phi1, phi2, arm1, arm2, az1, az2):
    """
    Integrate phase space, accounting for the fact that the phi+arm range is limited to [0,pi]
    """
    
    phi1, phi2, arm1, arm2, az1, az2 = np.broadcast_arrays(phi1, phi2, arm1, arm2, az1, az2)

    # Handle cases in between the physical boundaries
    # Integrate excluding unphysical corners
    # Remove unphysical rectangles
    arm1 = np.choose((arm1 < -phi2)        & (-phi2 < arm2),     [arm1, -phi2])
    arm2 = np.choose((arm1 < np.pi - phi1) & (np.pi-phi1 < arm2), [arm2, np.pi - phi1])    

    phi1 = np.choose((phi1 < -arm2)        & (-arm2 < phi2),     [phi1, -arm2])
    phi2 = np.choose((phi1 < np.pi - arm1) & (np.pi-arm1 < phi2), [phi2, np.pi - arm1])
    
    integral_rect = (az2-az1) * (-np.sin(arm1+phi1)+np.sin(arm2+phi1)+np.sin(arm1+phi2)-np.sin(arm2+phi2))

    # Remove unphysical corners (triangles or trapezoids)
    # Note the (phi1 + arm1) and (phi2 + arm2) masks in front

    # Lower left corner (low phi, low arm)
    # Integrate[Sin[phi+arm],{phi,phi1,phi2},{arm,arm1, -phi}]//FullSimplify
    phil = np.maximum(-arm2, phi1)
    phih = np.minimum(-arm1, phi2)
    unphys_lowerleft_integral = -phih+phil+np.sin(arm1+phih)-np.sin(arm1+phil)
    unphys_lowerleft_integral *= (phil + arm1 < 0)
    integral = integral_rect - (az2-az1) * unphys_lowerleft_integral

    # Upper right corner (high phi, high arm)
    # Integrate[Sin[phi+arm],{phi,phi1,phi2}, {arm, \[Pi]-phi, arm2}]//FullSimplify
    phil = np.maximum(np.pi - arm2, phi1)
    phih = np.minimum(np.pi - arm1, phi2)
    unphys_upperright_integral = phil-phih+np.sin(arm2+phil)-np.sin(arm2+phih)
    unphys_upperright_integral *= (phih + arm2 > np.pi)
    integral -= (az2-az1) * unphys_upperright_integral

    # Handle fully physical or fully unphysical
    fully_phys = (phi1 + arm1 >= 0) & (phi2 + arm2 <= np.pi)
    fully_unphys = (phi2 + arm2 <= 0) | (phi1 + arm1 >= np.pi)

    # Mathematica: Integrate[Sin[phi+arm], {phi,phi1,phi2} , {arm,arm1,arm2}]//FullSimplify
    integral_full = (az2-az1) * (-np.sin(arm1+phi1)+np.sin(arm2+phi1)+np.sin(arm1+phi2)-np.sin(arm2+phi2))
    
    if integral.ndim == 0:
        if fully_phys:
            return integral
        if fully_unphys:
            return 0
    else:
        integral[fully_phys] = integral_full[fully_phys]
        integral[fully_unphys] = 0
    
    return integral

### Plots and checks
angres = np.deg2rad(3) # Angular resolution. Will bin finely around Compton cone

phi_axis = Axis(np.linspace(0,np.pi,18+1))
arm_axis = Axis(np.concatenate((np.linspace(-np.pi, -3*angres, 12+1)[:-1], 
                                np.linspace(-3*angres, 3*angres, 13+1), 
                                np.linspace(3*angres, np.pi, 12+1)[1:])))
az_axis = Axis(np.linspace(-np.pi,np.pi,36+1))

phi_edges_mesh, arm_edges_mesh, az_edges_mesh = np.meshgrid(phi_axis.edges, arm_axis.edges, az_axis.edges, indexing = 'ij')

phase_space_rsp = phi_arm_az_phase_space(phi_edges_mesh[:-1, :-1, :-1], 
                                  phi_edges_mesh[ 1:, :-1, :-1], 
                                  arm_edges_mesh[:-1, :-1, :-1],
                                  arm_edges_mesh[:-1, 1:,  :-1:],
                                   az_edges_mesh[:-1, :-1, :-1],
                                   az_edges_mesh[:-1, :-1,  1:])

# Phase space within bin
# This is what the analysis needs. 
# It's bin dependent, as it should:
# fine binning results in low values
h_ps = Histogram([arm_axis, phi_axis], contents = np.sum(phase_space_rsp, axis = 2).transpose())

ax,plot = h_ps.plot()

x = np.linspace(-np.pi, np.pi, 100)
ax.plot(x, -x)
ax.plot(x, np.pi-x)

ax.set_ylim(0,np.pi)

ax.set_ylabel(r"$\phi$")
ax.set_xlabel(r"$\theta_{ARM}$")

ax.set_title("Phase space within bin")

# Phase space within bin per bin area
# Mostly only for visualization
bin_area = ((phi_edges_mesh[1:,  :-1, :-1]  - phi_edges_mesh[:-1, :-1, :-1]) * 
            (arm_edges_mesh[:-1, 1:,  :-1:] - arm_edges_mesh[:-1, :-1, :-1]) * 
             (az_edges_mesh[:-1, :-1,  1:]  -  az_edges_mesh[:-1, :-1, :-1]))

h_ps_diff = h_ps / np.sum(bin_area, axis = 2).transpose() 

ax,plot = h_ps_diff.plot()
x = np.linspace(-np.pi, np.pi, 100)
ax.plot(x, -x)
ax.plot(x, np.pi-x)

ax.set_ylim(0,np.pi)

ax.set_ylabel(r"$\phi$")
ax.set_xlabel(r"$\theta_{ARM}$")

ax.set_title("Phase space within bin per bin area")

# Should be 2*np.pi = 6.283185307179586 (az area = 2*pi, phi area = pi, and phi+arm (psichi radius) area = 2)
np.sum(h_ps/2/np.pi)