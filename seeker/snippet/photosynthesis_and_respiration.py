#date: 2024-05-30T17:08:48Z
#url: https://api.github.com/gists/db19ca1676cc31a699126f85357f603a
#owner: https://api.github.com/users/runck014

'''
Collaborative editing of C3 Photosynthesis sub-routine from https://github.com/Seji-jam/Generic-Object-Oriented-Plant-Model/blob/main/v3-0-OOP-operational/Leaf.py.

Refactored by ChatGPT, then collaboratively edited by Sajad, Diane, and Bryan. Simplified to only photosynthesis, moving dark respiration to another function.
'''

import numpy as np
import math
from respiration import respiration

# Constants
Activation_Energy_VCMAX = 65330  # Energy of activation for VCMAX (J/mol)
Activation_Energy_Jmax = 43790  # Energy of activation for Jmax (J/mol)
Entropy_Term_JT_Equation = 650  # Entropy term in JT equation (J/mol/K)
Deactivation_Energy_Jmax = 200000  # Energy of deactivation for Jmax (J/mol)
Maximum_Electron_Transport_Efficiency = 0.85  # Maximum electron transport efficiency of PS II
Protons_For_ATP_Synthesis = 3  # Number of protons required to synthesize 1 ATP
O2_Concentration = 210  # Oxygen concentration (mmol/mol)


def respiration(Leaf_Temp, Leaf_N_Content, Respiration_LeafN_Slope):
    """
    Calculate the dark respiration rate of a leaf.

    Parameters:
    - Leaf_Temp: Leaf temperature in °C.
    - Leaf_N_Content: Nitrogen content of the leaf.
    - Respiration_LeafN_Slope: Slope of respiration rate versus leaf nitrogen content.

    Returns:
    - Dark_Respiration: Dark respiration rate of the leaf.
    """
    # Temperature adjustment factor for respiration
    Respiration_Temperature_Effect = 0.0575 * math.exp(0.0693 * Leaf_Temp)

    # Adjusted respiration rate based on leaf nitrogen content
    Dark_Respiration = Respiration_LeafN_Slope * Respiration_Temperature_Effect * Leaf_N_Content

    return Dark_Respiration



def photosynthesis(Absorbed_PAR, Leaf_Temp, Intercellular_CO2, Leaf_N_Content, Vcmax_LeafN_Slope, Jmax_LeafN_Slope, Photosynthetic_Light_Response_Factor, Respiration_LeafN_Slope):
    """
    Calculate the net photosynthesis rate of a C3 plant leaf.

    Parameters:
    - Absorbed_PAR: Photosynthetically Active Radiation absorbed by the leaf.
    - Leaf_Temp: Leaf temperature in °C.
    - Intercellular_CO2: Intercellular CO2 concentration in μmol mol-1.
    - Leaf_N_Content: Nitrogen content of the leaf.
    - Vcmax_LeafN_Slope: Slope of Vcmax versus leaf nitrogen content.
    - Jmax_LeafN_Slope: Slope of Jmax versus leaf nitrogen content.
    - Photosynthetic_Light_Response_Factor: Light response factor for photosynthesis.
    - Respiration_LeafN_Slope: Slope of respiration rate versus leaf nitrogen content.

    Returns:
    - Net_Photosynthesis: Net photosynthesis rate of the leaf.
    """

    # Calculate dark respiration
    Dark_Respiration = respiration(Leaf_Temp, Leaf_N_Content, Respiration_LeafN_Slope)

    # Temperature adjustment factors
    temp_factor = 1. / 298. - 1. / (Leaf_Temp + 273.)
    Carboxylation_Temperature_Effect = math.exp(temp_factor * Activation_Energy_VCMAX / 8.314)
    Electron_Transport_Temperature_Effect = (math.exp(temp_factor * Activation_Energy_Jmax / 8.314) * 
        (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - Deactivation_Energy_Jmax / 298. / 8.314)) /
        (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - 1. / (Leaf_Temp + 273.) * Deactivation_Energy_Jmax / 8.314)))

    # Adjusted VCMAX and JMAX based on leaf nitrogen content
    Adjusted_VCMAX = Vcmax_LeafN_Slope * Carboxylation_Temperature_Effect * Leaf_N_Content
    Adjusted_JMAX = Jmax_LeafN_Slope * Electron_Transport_Temperature_Effect * Leaf_N_Content

    # Conversion of absorbed PAR to photon flux density
    Photon_Flux_Density = 4.56 * Absorbed_PAR  # Conversion factor to μmol m-2 s-1

    # Michaelis-Menten constants adjusted for temperature
    KMC = 404.9 * math.exp(temp_factor * 79430 / 8.314)
    KMO = 278.4 * math.exp(temp_factor * 36380 / 8.314)

    # CO2 compensation point without dark respiration
    CO2_Compensation_No_Respiration = 0.5 * math.exp(-3.3801 + 5220. / (Leaf_Temp + 273.) / 8.314) * O2_Concentration * KMC / KMO

    # Electron transport rate in response to absorbed PAR photon flux
    Quantum_Efficiency_Adjustment = (1 - 0) / (1 + (1 - 0) / Maximum_Electron_Transport_Efficiency)
    Electron_Transport_Ratio = Quantum_Efficiency_Adjustment * Photon_Flux_Density / max(1E-10, Adjusted_JMAX)
    Adjusted_Electron_Transport_Rate = Adjusted_JMAX * (1 + Electron_Transport_Ratio - ((1 + Electron_Transport_Ratio)**2 - 4 * Electron_Transport_Ratio * Photosynthetic_Light_Response_Factor)**0.5) / 2 / Photosynthetic_Light_Response_Factor

    # Carboxylation rates limited by Rubisco activity and electron transport
    Carboxylation_Rate_Rubisco_Limited = Adjusted_VCMAX * Intercellular_CO2 / (Intercellular_CO2 + KMC * (O2_Concentration / KMO + 1.))
    Carboxylation_Rate_Electron_Transport_Limited = Adjusted_Electron_Transport_Rate * Intercellular_CO2 * (2 + 0 - 0) / Protons_For_ATP_Synthesis / (0 + 3 * Intercellular_CO2 + 7 * CO2_Compensation_No_Respiration) / (1 - 0)

    # Gross rate of leaf photosynthesis
    Photosynthesis_Efficiency = (1 - CO2_Compensation_No_Respiration / Intercellular_CO2) * min(Carboxylation_Rate_Rubisco_Limited, Carboxylation_Rate_Electron_Transport_Limited)
    Gross_Leaf_Photosynthesis = max(1E-10, (1E-6) * 44 * Photosynthesis_Efficiency)

    # Net photosynthesis
    Net_Photosynthesis = Gross_Leaf_Photosynthesis - Dark_Respiration

    return Net_Photosynthesis

# Example usage
Absorbed_PAR = 500  # Example absorbed PAR
Leaf_Temp = 25  # Example leaf temperature
Intercellular_CO2 = 200  # Example intercellular CO2 concentration
Leaf_N_Content = 2.0  # Example leaf nitrogen content
Vcmax_LeafN_Slope = 50  # Example Vcmax slope
Jmax_LeafN_Slope = 100  # Example Jmax slope
Photosynthetic_Light_Response_Factor = 0.7  # Example light response factor
Respiration_LeafN_Slope = 0.02  # Example respiration slope

net_photosynthesis = photosynthesis(Absorbed_PAR, Leaf_Temp, Intercellular_CO2, Leaf_N_Content, Vcmax_LeafN_Slope, Jmax_LeafN_Slope, Photosynthetic_Light_Response_Factor, Respiration_LeafN_Slope)
print(f"Net Photosynthesis: {net_photosynthesis}")