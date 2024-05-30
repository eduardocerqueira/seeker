#date: 2024-05-30T16:58:51Z
#url: https://api.github.com/gists/eb59779d5342693378e390fc4c0e4852
#owner: https://api.github.com/users/runck014

'''
Collaborative editing of C3 Photosynthesis sub-routine from https://github.com/Seji-jam/Generic-Object-Oriented-Plant-Model/blob/main/v3-0-OOP-operational/Leaf.py.

Refactored by ChatGPT, the collaboratively edited by Sajad, Diane, and Bryan. Simplified to only photosynthesis, moving dark respiration to another function.
'''

import numpy as np
import math

# Constants
Activation_Energy_Dark_Respiration = 46390  # Energy of activation for dark respiration (J/mol)
Dark_Respiration_VCMAX_Ratio_25C = 0.0089  # Dark respiration to VCMAX ratio at 25°C

def respiration(Leaf_Temp, Vcmax_LeafN_Slope, Leaf_N_Content):
    """
    Calculate the dark respiration rate of a C3 plant leaf.
    
    Parameters:
    - Leaf_Temp: Leaf temperature in °C.
    - Vcmax_LeafN_Slope: Slope of Vcmax versus leaf nitrogen content.
    - Leaf_N_Content: Nitrogen content of the leaf.
    
    Returns:
    - Leaf_Dark_Respiration: Rate of leaf dark respiration.
    """
    
    # Temperature adjustment factor for dark respiration
    temp_factor = 1. / 298. - 1. / (Leaf_Temp + 273.)
    Temperature_Effect = math.exp(temp_factor * Activation_Energy_Dark_Respiration / 8.314)
    
    # Adjusted VCMAX based on leaf nitrogen content
    Adjusted_VCMAX = Vcmax_LeafN_Slope * Leaf_N_Content
    
    # Rate of leaf dark respiration
    Leaf_Dark_Respiration = (1E-6) * 44 * Dark_Respiration_VCMAX_Ratio_25C * Adjusted_VCMAX * Temperature_Effect
    
    return Leaf_Dark_Respiration