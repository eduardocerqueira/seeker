#date: 2025-02-26T17:11:04Z
#url: https://api.github.com/gists/cfb8abf0b49cb6e32b7a93c2ed9a208e
#owner: https://api.github.com/users/jfrobbins

import numpy as np


def evm_percent_to_db(evm_percent):
    """
    Convert EVM from percentage to dB.

    Parameters:
        evm_percent (float): EVM in percent.

    Returns:
        float: EVM in dB.
    """
    return 20 * np.log10(evm_percent / 100)


def evm_db_to_percent(evm_db):
    """
    Convert EVM from dB to percentage.

    Parameters:
        evm_db (float): EVM in dB.

    Returns:
        float: EVM in percent.
    """
    return 100 * 10 ** (evm_db / 20)


def evm_embed_pct(evm1_pct, evm2_pct):
    """
    Adds evm2 to evm1 (both in percent) using quadrature addition.

    Parameters:
        evm1_pct (float): Measured EVM in percent.
        evm2_pct (float): EVM in percent to add.

    Returns:
        float: Embedded EVM in percent.
    """
    return np.sqrt(np.square(evm1_pct) + np.square(evm2_pct))


def evm_embed_db(evm1_dB, evm2_to_embed_dB):
    """
    Adds evm2 to evm1 (both in dB) by converting to percent, performing quadrature addition, and converting back.

    Parameters:
        evm1_dB (float): Measured EVM in dB.
        evm2_to_embed_dB (float): EVM in dB to add.

    Returns:
        float: Embedded EVM in dB.
    """
    evm1_pct = evm_db_to_percent(evm1_dB)
    evm2_pct = evm_db_to_percent(evm2_to_embed_dB)

    evm_embedded_pct = evm_embed_pct(evm1_pct, evm2_pct)
    return evm_percent_to_db(evm_embedded_pct)


def evm_deembed_pct(evm1_pct, evm2_pct, minSystemEvmFloor_pct):
    """
    De-embeds evm2 from evm1 (both in percent) using quadrature subtraction.
    If evm2_pct > evm1_pct for any element, that element simply returns evm1_pct.
    Otherwise, the de-embedded value is computed as:
      result = evm_embed_pct( sqrt(evm1_pct^2 - evm2_pct^2), minSystemEvmFloor_pct )
    
    Parameters:
        evm1_pct (float or array-like): Measured EVM in percent.
        evm2_pct (float or array-like): System floor EVM in percent to subtract.
        minSystemEvmFloor_pct (float or array-like): EVM floor in percent to add back in.
    
    Returns:
        float or ndarray: De-embedded EVM in percent.
    """
    # Calculate the difference using quadrature subtraction (ensure non-negative)
    diff = np.sqrt(np.maximum(np.square(evm1_pct) - np.square(evm2_pct), 0))
    
    # Compute the result by adding in the minimum system floor in quadrature
    result = evm_embed_pct(diff, minSystemEvmFloor_pct)
    
    # If the system floor is greater than the measured value, just return the measured value.
    return np.where(evm2_pct > evm1_pct, evm1_pct, result)



def evm_deembed_db(evm1_dB, evm2_to_deembed_dB, minSystemEvmFloor_db=-60.0):
    """
    De-embeds evm2 from evm1 (both in dB) by converting to percent, performing quadrature subtraction,
    and converting the result back to dB. Optionally adds back in a default floor.

    Parameters:
        evm1_dB (float): Measured EVM in dB.
        evm2_to_deembed_dB (float): System floor EVM in dB to subtract.
        minSystemEvmFloor_db (float): System EVM floor in dB to add back in, or use as min.

    Returns:
        float: De-embedded EVM in dB.
    """
    evm1_pct = evm_db_to_percent(evm1_dB)
    evm2_pct = evm_db_to_percent(evm2_to_deembed_dB)

    evm_de_pct = evm_deembed_pct(evm1_pct, evm2_pct, evm_db_to_percent(minSystemEvmFloor_db))
    return evm_percent_to_db(evm_de_pct)


    
# Example usage:
if __name__ == "__main__":
    print("=== Conversions Between dB and Percent ===")
    # Convert EVM from dB to percent
    evm_db_value = -47  # example value in dB
    evm_percent_value = evm_db_to_percent(evm_db_value)
    print(f"{evm_db_value} dB EVM is equivalent to {evm_percent_value:.2f}%")

    # Convert EVM from percent to dB
    evm_percent_value = 1  # example value in percent
    evm_db_value = evm_percent_to_db(evm_percent_value)
    print(f"{evm_percent_value}% EVM is equivalent to {evm_db_value:.2f} dB")
    
    # More examples: Looping over multiple values
    test_dB_values = [-30, -35, -42, -55]
    for dB in test_dB_values:
        pct = evm_db_to_percent(dB)
        print(f"{dB} dB converts to {pct:.2f}% EVM")
        
    test_percent_values = [0.5, 1, 2]
    for pct in test_percent_values:
        dB = evm_percent_to_db(pct)
        print(f"{pct}% EVM converts to {dB:.2f} dB")
    
    print("\n=== De-embedding Examples ===")
    # Example 1: De-embed in dB domain directly
    measured_evm_db = -45.2   # example measured EVM in dB
    system_floor_db = -47.0   # example system floor in dB
    deembedded_evm_db = evm_deembed_db(measured_evm_db, system_floor_db)
    print(f"Measured EVM: {measured_evm_db:.2f} dB with system floor {system_floor_db:.2f} dB")
    print(f"De-embedded EVM: {deembedded_evm_db:.2f} dB")
    
    # Example 2: De-embed in dB domain directly with different values
    measured_evm_db = -38.99   # example measured EVM in dB
    system_floor_db = -49.0    # example system floor in dB
    deembedded_evm_db = evm_deembed_db(measured_evm_db, system_floor_db)
    print(f"Measured EVM: {measured_evm_db:.2f} dB with system floor {system_floor_db:.2f} dB")
    print(f"De-embedded EVM: {deembedded_evm_db:.2f} dB")
    
    # Example 3: De-embed in dB domain directly with invalid values
    measured_evm_db = -48.99   # example measured EVM in dB
    system_floor_db = -45.0    # example system floor in dB
    deembedded_evm_db = evm_deembed_db(measured_evm_db, system_floor_db)
    print(f"Measured EVM: {measured_evm_db:.2f} dB with system floor {system_floor_db:.2f} dB")
    print(f"De-embedded EVM: {deembedded_evm_db:.2f} dB")    
