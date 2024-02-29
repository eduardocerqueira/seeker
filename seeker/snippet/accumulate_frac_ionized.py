#date: 2024-02-29T17:05:39Z
#url: https://api.github.com/gists/7a3b8ebf387b767eae6e1ce78cfb334f
#owner: https://api.github.com/users/ncook882

import numpy as np
from scipy.special import jv, jn_zeros
import scipy.constants

q_e = scipy.constants.e
m_e = scipy.constants.m_e
c = scipy.constants.c
eps = scipy.constants.epsilon_0


wavelength = 800.0e-9
omega = (2.0*np.pi*c)/wavelength
e_mec2 = 510998.9499961642 #511 keV

def accumulate_frac_ionized(E_max, R, dt=0.1e-15, r0=3.5):
    "Accumulate ionization and temeprature on a fixed r mesh"
    
    dr = 0.1
    Nr = round(R/dr)+1
    rs = np.linspace(-1.*R,1.*R,Nr)
    
    fi_total = np.zeros(len(rs))
    fi_instant = np.zeros(len(rs))
    Txl_total = np.zeros(len(rs))
    Txc_total = np.zeros(len(rs))
    ai_instant = np.zeros(len(rs))
    
    #start 1 FWHM away from peak
    FWHM = 40.0e-15
    #dt = 0.05e-15
    nt = round(2.*FWHM/dt)  
                          
    #times for history purposes
    ts = np.arange(nt)*dt-FWHM
    
    #print(ts)
    
    fi_history = np.zeros(len(ts))
    Txl_history = np.zeros(len(ts))
    Txl_added = np.zeros(len(ts))
    fi_added = np.zeros(len(ts))
    
    fi_offpeak = np.zeros(len(ts))
    fi_offpeak2 = np.zeros(len(ts))
    
    #compute prefactor
    eulers_number = 2.71828
    E_h = 5.13e11 #V/m
    Z = 1.0 # charge state of ionized particle
    U_ion = 15.4 # ionization potential in eV H2
    n_eff = Z / np.sqrt(U_ion / 13.6)
    a1 = 6.6e16 * (eulers_number / np.pi) * (Z**2.0 / n_eff**4.5)
    
    U_i = U_ion
    U_h = 13.6
    E_a = 5.13e11 #V/m E_h
    
    for ind,t in enumerate(ts):
        #compute radial field and normalized intensity
        E_L = np.abs(E_max * jv(0, rs/r0) * np.exp(-4.0*np.log(2.0)* (t/FWHM)**2.0))
        ai_instant = (q_e/(m_e*c*omega)) * E_L 
        
        #print(np.max(ai_instant))
        
        #compute ionization rate for time step
        b2 = (10.87 * (E_h / E_L) * (Z**3.0 / n_eff**4.0))**(2.0*n_eff - 1.5)
        c3 = np.exp(-(2.0/3.0) * (E_h / E_L) * (Z / n_eff)**3.0)
        adk_ionization_rate_per_sec = (a1 * b2 * c3)
        Wi = adk_ionization_rate_per_sec*dt
        Wi = np.clip(Wi,0,1)
        
        #update fraction ionized
        fi_instant = (1-fi_total)*Wi
        fi_total = fi_total + fi_instant
        
        fi_history[ind] = np.max(fi_total)
        fi_added[ind] = np.max(fi_instant)
        
        fi_offpeak[ind] = fi_total[182]
        fi_offpeak2[ind] = fi_total[164]
        
        #now update effective temp contributions
        off_peak_ionization = (U_h / U_i)**(3.0/4.0) * np.sqrt((3.0/2.0)*(E_L/E_a))
        Txl = 0.5*m_e*c**2.0*ai_instant**2.0 * off_peak_ionization**2.0
        Txc = 0.5*m_e*c**2.0*ai_instant**2.0 * np.ones(np.shape(off_peak_ionization))
        
        #add kinetic energy contribution weighted by ionization fraction
        Txl_total = Txl_total + Txl*fi_instant
        Txc_total = Txc_total + Txc*fi_instant
        
        Txl_history[ind] = np.max(Txl_total)
        Txl_added[ind] = np.max(Txl*fi_instant)

    return fi_total, Txl_total, Txc_total, fi_history, Txl_history, ts, fi_offpeak, fi_offpeak2, rs