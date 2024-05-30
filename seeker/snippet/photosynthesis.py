#date: 2024-05-30T16:48:54Z
#url: https://api.github.com/gists/e1db2c07d3f1c4b166da67905466a2f2
#owner: https://api.github.com/users/runck014

'''
Collaborative editing of C3 Photosynthesis sub-routine from https://github.com/Seji-jam/Generic-Object-Oriented-Plant-Model/blob/main/v3-0-OOP-operational/Leaf.py.


Refactored by ChatGPT, the collaboratively edited by Sajad, Diane, and Bryan. Simplified to only photosynthesis, moving dark respiration to another function.

'''

import numpy as np
import math

# Constants
O2_Concentration = 210  # Oxygen concentration (mmol/mol)
Activation_Energy_KMC = 79430  # Energy of activation for KMC (J/mol)
Activation_Energy_KMO = 36380  # Energy of activation for KMO (J/mol)
Activation_Energy_VCMAX = 65330  # Energy of activation for VCMAX (J/mol)
Activation_Energy_Jmax = 200000  # Energy of deactivation for JMAX (J/mol)
Entropy_Term_JT_Equation = 650  # Entropy term in JT equation (J/mol/K)
Protons_For_ATP_Synthesis = 3  # Number of protons required to synthesize 1 ATP
Maximum_Electron_Transport_Efficiency = 0.85  # Maximum electron transport efficiency of PS II

def photosynthesis(Absorbed_PAR, Leaf_Temp, Intercellular_CO2, Leaf_N_Content, Vcmax_LeafN_Slope, Jmax_LeafN_Slope, Photosynthetic_Light_Response_Factor):
    """
    Calculate the potential photosynthesis rate of a C3 plant leaf.
    
    Parameters:
    - Absorbed_PAR: Absorbed photosynthetically active radiation (PAR) by the leaf.
    - Leaf_Temp: Leaf temperature in °C.
    - Intercellular_CO2: Intercellular CO2 concentration in the leaf.
    - Leaf_N_Content: Nitrogen content of the leaf.
    - Vcmax_LeafN_Slope: Slope of Vcmax versus leaf nitrogen content.
    - Jmax_LeafN_Slope: Slope of Jmax versus leaf nitrogen content.
    - Photosynthetic_Light_Response_Factor: Photosynthetic light response factor.
    
    Returns:
    - Gross_Leaf_Photosynthesis: Gross rate of leaf photosynthesis.
    """
    
    # Constants for Michaelis-Menten constants for CO2 and O2 at 25°C for C3 plants
    Michaelis_Menten_CO2_25C = 404.9
    Michaelis_Menten_O2_25C = 278.4
    
    # Conversion of absorbed PAR to photon flux density
    Photon_Flux_Density = 4.56 * Absorbed_PAR  # Conversion factor to umol/m^2/s
    
    # Adjusting Michaelis-Menten constants for CO2 and O2 with temperature
    temp_factor = 1. / 298. - 1. / (Leaf_Temp + 273.)
    CO2_Michaelis_Menten_Temp_Adjusted = Michaelis_Menten_CO2_25C * math.exp(temp_factor * Activation_Energy_KMC / 8.314)
    O2_Michaelis_Menten_Temp_Adjusted = Michaelis_Menten_O2_25C * math.exp(temp_factor * Activation_Energy_KMO / 8.314)
    
    # CO2 compensation point without dark respiration
    CO2_Compensation_No_Respiration = 0.5 * math.exp(-3.3801 + 5220. / (Leaf_Temp + 273.) / 8.314) * O2_Concentration * CO2_Michaelis_Menten_Temp_Adjusted / O2_Michaelis_Menten_Temp_Adjusted
    
    # Temperature effects on carboxylation and electron transport
    Carboxylation_Temperature_Effect = math.exp(temp_factor * Activation_Energy_VCMAX / 8.314)
    Electron_Transport_Temperature_Effect = math.exp(temp_factor * Activation_Energy_Jmax / 8.314) * \
        (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - Activation_Energy_Jmax / 298. / 8.314)) / \
        (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - 1. / (Leaf_Temp + 273.) * Activation_Energy_Jmax / 8.314))
    
    # Adjusted VCMAX and JMAX based on leaf nitrogen content
    Adjusted_VCMAX = Vcmax_LeafN_Slope * Carboxylation_Temperature_Effect * Leaf_N_Content
    Adjusted_JMAX = Jmax_LeafN_Slope * Electron_Transport_Temperature_Effect * Leaf_N_Content
    
    # Electron transport rate in response to absorbed PAR photon flux
    Quantum_Efficiency_Adjustment = 1 / (1 + (1 / Maximum_Electron_Transport_Efficiency))
    Electron_Transport_Ratio = Quantum_Efficiency_Adjustment * Photon_Flux_Density / max(1E-10, Adjusted_JMAX)
    Adjusted_Electron_Transport_Rate = Adjusted_JMAX * (1 + Electron_Transport_Ratio - ((1 + Electron_Transport_Ratio)**2 - 4 * Electron_Transport_Ratio * Photosynthetic_Light_Response_Factor)**0.5) / 2 / Photosynthetic_Light_Response_Factor
    
    # Carboxylation rates limited by Rubisco activity and electron transport
    Carboxylation_Rate_Rubisco_Limited = Adjusted_VCMAX * Intercellular_CO2 / (Intercellular_CO2 + CO2_Michaelis_Menten_Temp_Adjusted * (O2_Concentration / O2_Michaelis_Menten_Temp_Adjusted + 1))
    Carboxylation_Rate_Electron_Transport_Limited = Adjusted_Electron_Transport_Rate * Intercellular_CO2 / (CO2_Michaelis_Menten_Temp_Adjusted + Intercellular_CO2)
    
    # Gross rate of leaf photosynthesis
    Photosynthesis_Efficiency = (1 - CO2_Compensation_No_Respiration / Intercellular_CO2) * min(Carboxylation_Rate_Rubisco_Limited, Carboxylation_Rate_Electron_Transport_Limited)
    Gross_Leaf_Photosynthesis = max(1E-10, (1E-6) * 44 * Photosynthesis_Efficiency)
    
    return Gross_Leaf_Photosynthesis