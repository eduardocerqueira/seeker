#date: 2025-07-09T17:07:51Z
#url: https://api.github.com/gists/614cce365e05397b64f16d9e903e1499
#owner: https://api.github.com/users/gab-gash

import math
from fractions import Fraction

# --- UTILITY FUNCTIONS FOR MONZO AND FACTORIZATION ---

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
]


def prime_factorize_fraction(number_fraction, primes_list):
    """
    Factorizes a fraction (or integer) into its prime components.
    Returns a Monzo (list of exponents) and a boolean indicating if the factorization is complete.
    """
    monzo = [Fraction(0, 1)] * len(primes_list)
    is_completely_factorized = True

    num = number_fraction.numerator
    den = number_fraction.denominator

    def _factorize_part(value, is_numerator):
        nonlocal is_completely_factorized
        temp_val = value
        for i, p in enumerate(primes_list):
            if temp_val == 1:
                break

            while temp_val % p == 0:
                if is_numerator:
                    monzo[i] += Fraction(1, 1)
                else:
                    monzo[i] -= Fraction(1, 1)
                temp_val //= p
        
        if temp_val != 1:
            is_completely_factorized = False # Remaining factor not in PRIMES
        return temp_val

    _factorize_part(num, True)
    _factorize_part(den, False)

    return monzo, is_completely_factorized


def calculate_cents(value):
    """Calculates cents for a given value."""
    if value <= 0:
        return 0.0
    return 1200 * math.log2(float(value))

def normalize_interval(val_original, monzo, base_a, primes_list, is_monzo_valid):
    """
    Normalizes a Monzo with respect to a base 'a' (the period),
    such that the resulting interval is >= 1 and < base_a.
    This eliminates negative cents.

    Returns:
    - normalized_monzo: the normalized Monzo (dummy if is_monzo_valid is False)
    - normalized_val: the numerically normalized value (derived from monzo if valid, or from val_original if invalid)
    - is_normalized_monzo_valid: indicates if the normalized monzo is significant
    """

    if base_a <= 0: # Handle invalid base_a
        return [Fraction(0,1)] * len(primes_list), val_original, False # Cannot normalize, return original val

    base_a_cents = calculate_cents(base_a)
    if base_a_cents == 0 and base_a != 1: # Base is 0 or negative but not 1
        return [Fraction(0,1)] * len(primes_list), val_original, False # Cannot normalize

    if not is_monzo_valid:
        # If original monzo is not valid (e.g., non-prime-limit number),
        # we still numerically normalize the value and calculate cents from it.
        # The monzo itself will remain a dummy and marked as invalid for display.
        current_val_for_normalization = val_original
        interval_cents = calculate_cents(current_val_for_normalization)

        if base_a_cents == 0: # Base is 1
            normalized_val = 1.0
        else:
            num_periods_to_shift = math.floor(interval_cents / base_a_cents)
            normalized_val = val_original / (base_a ** num_periods_to_shift)
        
        return [Fraction(0,1)] * len(primes_list), normalized_val, False
    else:
        # If original monzo is valid, proceed with monzo-based normalization
        val_from_monzo = 1.0
        for i, exp in enumerate(monzo):
            val_from_monzo *= (primes_list[i] ** float(exp))
        
        if val_from_monzo <= 0:
            return [Fraction(0,1)] * len(primes_list), 1.0, True # Should not happen with valid monzos usually

        interval_cents = calculate_cents(val_from_monzo)

        if base_a_cents == 0: # Base is 1
            normalized_monzo = [Fraction(0,1)] * len(primes_list)
            normalized_val = 1.0
        else:
            base_a_fraction = Fraction(base_a, 1)
            base_a_monzo, _ = prime_factorize_fraction(base_a_fraction, primes_list) 
            
            num_periods_to_shift = math.floor(interval_cents / base_a_cents)

            normalized_monzo = list(monzo) # Copy for not modifying the original
            for i in range(len(normalized_monzo)):
                normalized_monzo[i] -= base_a_monzo[i] * num_periods_to_shift

            normalized_val = 1.0
            for i, exp in enumerate(normalized_monzo):
                normalized_val *= (primes_list[i] ** float(exp))
        
        return normalized_monzo, normalized_val, True


def calculate_b_eda_approximation(target_value, base_a, b):
    """
    Calculates the b-EDa approximation for a given target value,
    and returns the x/b fraction, the error in cents, and the new error Err(a'b).
    """
    if base_a <= 0 or b <= 0 or target_value <= 0: # Defensive check
        return "Invalid_Params", 0.0, 0.0

    target_cents = calculate_cents(target_value) # Use safe calculate_cents

    cents_per_base_a = calculate_cents(float(base_a)) # Use safe calculate_cents
    if cents_per_base_a == 0: # Base is 1
        return f"0\\{b}", 0.0, 0.0

    cents_per_tick = cents_per_base_a / b

    num_ticks = round(target_cents / cents_per_tick)
    
    # Calculate the approximated value (p/X) from num_ticks
    # The value approximated by b-EDa is base_a^(num_ticks / b)
    approx_value = base_a ** (num_ticks / b)

    error_cents = calculate_cents(approx_value) - target_cents # Err(c) = cents of approx - cents of target

    # Calculate Err(a'b) = (100 * b / log2(base_a)) * log2(approx_value / target_value)
    # This ensures Err(a'b) has the same sign as error_cents
    if base_a == 1 or math.log2(base_a) == 0: # Avoid division by zero, use cents for consistency
        new_error_ab = 0.0
    elif target_value <= 0 or approx_value <= 0: # Logarithm arguments must be positive
        new_error_ab = 0.0
    else:
        new_error_ab = (100 * b / math.log2(base_a)) * math.log2(approx_value / target_value)

    return f"{int(num_ticks)}\\{b}", error_cents, new_error_ab


def format_monzo_factorization(monzo_list, primes, is_completely_factorized):
    """
    Formats a Monzo list (Fraction exponents) into a readable string
    showing prime factorization.
    """
    if not is_completely_factorized:
        return "N.C."

    parts = []
    for i, exponent in enumerate(monzo_list):
        if exponent != Fraction(0, 1):
            prime = primes[i]
            if exponent.denominator == 1:
                if exponent.numerator == 1:
                    parts.append(f"{prime}")
                else:
                    parts.append(f"{prime}^{exponent.numerator}")
            else:
                if exponent.numerator == 1:
                    parts.append(f"{prime}^(1/{exponent.denominator})")
                else:
                    parts.append(f"{prime}^({exponent.numerator}/{exponent.denominator})")
    return " * ".join(parts) if parts else "1"


def format_monzo_vector_angle_brackets(monzo_list, is_completely_factorized):
    """
    Formats a Monzo list (Fraction exponents) into a compact vector string,
    truncating trailing zeros for readability, with angle brackets.
    """
    if not is_completely_factorized:
        return "N.C."

    last_nonzero_index = -1
    for i in range(len(monzo_list) - 1, -1, -1):
        if monzo_list[i] != Fraction(0, 1):
            last_nonzero_index = i
            break
    
    if last_nonzero_index == -1:
        return "[0⟩" # Format [0⟩
    
    formatted_exponents = []
    for i in range(last_nonzero_index + 1):
        exponent = monzo_list[i]
        if exponent.denominator == 1:
            formatted_exponents.append(str(exponent.numerator))
        else:
            formatted_exponents.append(f"{exponent.numerator}/{exponent.denominator}")
    
    return f"[{' '.join(formatted_exponents)}⟩" # Format [x y z⟩


def parse_value_input(input_str):
    """
    Parses the input string for a numerical value (integers, fractions, expressions).
    Returns the numerical value (float) and the Monzo.
    """
    try:
        local_dict = {'Fraction': Fraction, 'math': math}
        
        # Handles power operator '^' and replaces it with Python's '**'
        if '^' in input_str and '**' not in input_str:
            base_exp_parts = input_str.split('^')
            if len(base_exp_parts) == 2:
                base_str = base_exp_parts[0].strip()
                exp_str = base_exp_parts[1].strip()
                # Handles cases like 3^(1/4)
                if exp_str.startswith('(') and exp_str.endswith(')'):
                    exp_str = exp_str[1:-1] # Removes parentheses for eval
                input_str = f"({base_str})**({exp_str})"

        val_eval = eval(input_str, {"__builtins__": None}, local_dict)

        if not isinstance(val_eval, (int, float, Fraction)):
            print(f"Error: Input '{input_str}' produced an unsupported type ({type(val_eval)}).")
            return None, None, None
            
        if float(val_eval) <= 0: # Ensure interval is positive
            print(f"Error: Interval value '{input_str}' must be strictly positive (> 0).")
            return None, None, None

        if isinstance(val_eval, int):
            fraction_val = Fraction(val_eval, 1)
        elif isinstance(val_eval, float):
            # Tries to convert float to fraction, limiting denominator to avoid huge fractions from imprecise floats
            fraction_val = Fraction(val_eval).limit_denominator(10**6) 
        elif isinstance(val_eval, Fraction):
            fraction_val = val_eval

        monzo, is_completely_factorized = prime_factorize_fraction(fraction_val, PRIMES)
        
        return float(fraction_val), monzo, is_completely_factorized

    except (ValueError, SyntaxError, NameError, TypeError) as e:
        print(f"Input error for '{input_str}': {e}. Ensure you use a valid format.")
        return None, None, None

def get_monzo_from_user_input(current_input_mode_ref, i):
    """
    Prompts the user to enter Monzo exponents one by one.
    Returns a Monzo (list of Fractions) and its numerical value,
    or None, None if the user decides to finish/change mode.
    Also returns the new input mode if changed.
    """
    monzo_exponents = []
    
    print("\n--- Monzo Exponent Entry ---") # Keeps title clear for each new Monzo

    print("Type 'f' to finish Monzo entry.")
    print("At any time, you can type 'k' to cancel and exit, 'V' to switch to Numerical Value, 'D' to delete.") # Updated prompt
    
    for idx, prime in enumerate(PRIMES):
        while True:
            exp_str = input(f"Exponent for ({prime}): ").strip().lower()
            
            if exp_str == 'f':
                # Pad with zeros for remaining exponents
                monzo_exponents.extend([Fraction(0, 1)] * (len(PRIMES) - idx))
                # Print Monzo confirmation
                monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
                # Ensure calculated value is positive (Monzo mode doesn't prevent negatives by input, only by resulting value)
                if monzo_val_calculated <= 0:
                    print(f"Error: The entered Monzo results in a non-positive value ({monzo_val_calculated}). Please re-enter.")
                    # Clear current exponents to restart for this monzo or allow user to exit
                    monzo_exponents.clear() 
                    continue # Re-ask for current prime exponent, effectively restarting this monzo input
                print(f"--- Interval {i} has Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Format [X Y Z⟩
                return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True
            
            elif exp_str == 'k':
                print("Monzo entry canceled.")
                return None, None, 'k_terminate', None
            
            elif exp_str == 'v':
                current_input_mode_ref['mode'] = 'V'
                print("\nMode changed to **Numerical Value**.")
                print("Monzo entry canceled.") # If mode is changed, current Monzo is canceled
                return None, None, current_input_mode_ref.get('mode'), None 
            
            elif exp_str == 'm': # If user types 'm' while already in Monzo mode
                print("\nAlready in **Fractional Monzo** mode.")
                print("Monzo entry canceled.") # Cancel current Monzo and restart
                return None, None, current_input_mode_ref.get('mode'), None
            
            elif exp_str == 'd': # Option to delete during Monzo entry
                return None, None, 'd_delete', None

            try:
                if '/' in exp_str:
                    num, den = map(int, exp_str.split('/'))
                    if den == 0:
                        raise ValueError("Denominator cannot be zero.")
                    exponent = Fraction(num, den)
                else:
                    exponent = Fraction(int(exp_str), 1)
                monzo_exponents.append(exponent)
                break 
            except ValueError:
                print("Invalid input. Please enter an integer or a fraction (e.g., '1/2').")
    
    # If the loop finishes (all PRIMES have been entered)
    monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
    if monzo_val_calculated <= 0:
        print(f"Error: The entered Monzo results in a non-positive value ({monzo_val_calculated}). Entry canceled.")
        return None, None, current_input_mode_ref.get('mode'), False # Treat as invalid
    print(f"--- Interval {i} has Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Format [X Y Z⟩
    return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True


def calculate_monzo_value(monzo_exponents, primes_list):
    """Calculates the numerical value from a Monzo."""
    value = 1.0
    for i, exp in enumerate(monzo_exponents):
        if i < len(primes_list): 
            # For rational Monzo, exp can be an integer or fractional Fraction
            # If exp is a Fraction, float(exp) converts it to float, which also handles fractions.
            value *= (primes_list[i] ** float(exp))
    return value


def format_value_for_alignment(value_to_format, is_rational_monzo, is_completely_factorized, total_width_before_decimal, total_width_after_decimal):
    """
    Formats a numerical value for alignment.
    Generates a string in the form "decimal;fraction" if applicable, otherwise just "decimal".
    Aligns the decimal/fractional part for numbers.
    """
    # Ensure value_to_format is numeric for formatting
    if not isinstance(value_to_format, (int, float, Fraction)):
        return str(value_to_format) # Return as is if not numeric

    decimal_part_str = f"{float(value_to_format):.5f}" # Always convert to float for consistent decimal formatting
    
    fraction_part_str_formatted = ""
    # Only try to show fraction if it's rational and completely factorized
    if is_rational_monzo and is_completely_factorized:
        try:
            frac = Fraction(value_to_format).limit_denominator(10**6)
            # Only show fraction if it's not an exact integer (e.g., 2/1) or if user wants to explicitly show integer as fraction
            if frac.denominator != 1:
                fraction_part_str_formatted = f";{frac.numerator}/{frac.denominator}"
            elif frac.denominator == 1: # For integers, show like ;N if requested
                fraction_part_str_formatted = f";{frac.numerator}" # User wants integer as fraction too
        except (OverflowError, ZeroDivisionError, ValueError):
            pass # If conversion to Fraction fails, just use decimal

    # Split decimal part into integer and fractional components for alignment
    if '.' in decimal_part_str:
        int_part_dec, frac_part_dec = decimal_part_str.split('.')
        separator = "."
    else: # It's an integer or float without explicit decimal part
        int_part_dec = decimal_part_str
        frac_part_dec = "" 
        separator = ""
    
    # Pad integer part for alignment
    padded_int_part = f"{int_part_dec:>{total_width_before_decimal}}"
    
    # Pad fractional part for alignment
    # If no fractional part, pad with spaces to align with others that have it
    padded_frac_part = f"{frac_part_dec:<{total_width_after_decimal}}"

    # Combine parts
    if frac_part_dec or total_width_after_decimal > 0: # If there's a decimal part or we need to reserve space for it
        aligned_decimal_str = f"{padded_int_part}{separator}{padded_frac_part}"
    else: # Pure integer, no decimal point, just pad the integer part
        aligned_decimal_str = padded_int_part

    return f"{aligned_decimal_str}{fraction_part_str_formatted}"


def get_aligned_lengths(processed_intervals, current_b_chunk, min_b, max_b, current_a):
    """Calculates the maximum required widths to align columns."""
    # Minimum initial widths for headers
    col_widths = {
        "Monzo Norm.": len("Monzo Norm."), 
        "Value": len("Value"), 
        "Cents": len("Cents") 
    }
    
    max_val_before_sep = 0 # Max digits before '.' or '/'
    max_val_after_sep = 0  # Max digits after '.' or '/'
    max_val_full_string_len = len("Value") # For overall column width of 'Value' header

    max_cents_left = 0
    max_cents_right = 0
    
    # New variables for error_c and error_ab
    max_approx_num_len = 0 
    max_approx_den_len = 0 
    max_approx_err_c_left = 0
    max_approx_err_c_right = 0
    max_approx_err_ab_left = 0
    max_approx_err_ab_right = 0 

    for p_interval in processed_intervals:
        col_widths["Monzo Norm."] = max(col_widths["Monzo Norm."], len(p_interval['monzo_display']))

        # Calculate max lengths for Value column alignment
        val_to_check = p_interval['value'] # Use the actual numeric value for determining format
        is_rational_to_format = is_monzo_rational(p_interval['monzo_normalized_raw']) and p_interval['is_normalized_monzo_valid_for_display']

        # Determine decimal part length
        # Ensure val_to_check is numeric before formatting
        if isinstance(val_to_check, (float, int, Fraction)):
            decimal_part_str_temp = f"{float(val_to_check):.5f}"
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                max_val_before_sep = max(max_val_before_sep, len(int_p_dec))
                max_val_after_sep = max(max_val_after_sep, len(frac_p_dec))
            else:
                max_val_before_sep = max(max_val_before_sep, len(decimal_part_str_temp))
                max_val_after_sep = max(max_val_after_sep, 0)
        else: # Handle non-numeric values like "Error" for width calculation
            max_val_before_sep = max(max_val_before_sep, len(str(val_to_check)))
            max_val_after_sep = max(max_val_after_sep, 0)

        # Determine full formatted string length including optional fraction for overall column width
        temp_formatted_value_string = format_value_for_alignment(
            val_to_check, 
            is_rational_to_format, 
            p_interval['is_normalized_monzo_valid_for_display'], # Use true factorization state for the fraction part
            max_val_before_sep, # Use preliminary max_val_before_sep
            max_val_after_sep # Use preliminary max_val_after_sep
        )
        max_val_full_string_len = max(max_val_full_string_len, len(temp_formatted_value_string))


        cents_str_val = p_interval['display_cents'] # Use display_cents for width calculation
        cents_str_formatted = f"{cents_str_val:+.3f}" if isinstance(cents_str_val, (float, int)) else str(cents_str_val) # Defensive check

        if '.' in cents_str_formatted:
            cl, cr = cents_str_formatted.split('.')
            max_cents_left = max(max_cents_left, len(cl))
            max_cents_right = max(max_cents_right, len(cr))
        else:
            max_cents_left = max(max_cents_left, len(cents_str_formatted))

        for b_val_actual_idx in range(len(p_interval['all_approximations'])):
            # Correction: Calculate current 'b' index
            current_b_from_idx = min_b + b_val_actual_idx
            if current_b_from_idx in current_b_chunk: # Only for columns in the current chunk
                approx_fraction_str, error_c, error_ab = p_interval['all_approximations'][b_val_actual_idx]
                if approx_fraction_str != "Invalid_Params":
                    num_part, den_part = approx_fraction_str.split('\\')
                    
                    max_approx_num_len = max(max_approx_num_len, len(num_part))
                    max_approx_den_len = max(max_approx_den_len, len(den_part))

                    error_c_str = f"{error_c:+.3f}" if isinstance(error_c, (float, int)) else str(error_c) # Defensive check
                    if '.' in error_c_str:
                        err_cl, err_cr = error_c_str.split('.')
                        max_approx_err_c_left = max(max_approx_err_c_left, len(err_cl))
                        max_approx_err_c_right = max(max_approx_err_c_right, len(err_cr))
                    else:
                        max_approx_err_c_left = max(max_approx_err_c_left, len(error_c_str))

                    error_ab_str = f"{error_ab:+.3f}" if isinstance(error_ab, (float, int)) else str(error_ab) # Defensive check
                    if '.' in error_ab_str:
                        err_abl, err_abr = error_ab_str.split('.')
                        max_approx_err_ab_left = max(max_approx_err_ab_left, len(err_abl))
                        max_approx_err_ab_right = max(max_approx_err_ab_right, len(err_abr))
                    else:
                        max_approx_err_ab_left = max(max_approx_err_ab_left, len(error_ab_str))
                else: # Handle "Invalid_Params" for width calculation
                    max_approx_num_len = max(max_approx_num_len, len("Error")) # For "Error" string
                    max_approx_den_len = max(max_approx_den_len, 0)
                    max_approx_err_c_left = max(max_approx_err_c_left, 0)
                    max_approx_err_c_right = max(max_approx_err_c_right, 0)
                    max_approx_err_ab_left = max(max_approx_err_ab_left, 0)
                    max_approx_err_ab_right = max(max_approx_err_ab_right, 0)


    # Finalize Value column width based on separator alignment AND full string length
    col_widths["Value"] = max(max_val_full_string_len, len("Value")) 

    # Add extra space for good padding and decimal point/sign
    col_widths["Cents"] = max(col_widths["Cents"], max_cents_left + 1 + max_cents_right)

    # New approx_col_total_data_width calculation:
    # num\den + space + sign_c_digits_c.decimals_c + semicolon + sign_ab_digits_ab.decimals_ab + padding
    # +1 for '\', +1 for space, +1 for dot_c, +1 for semicolon, +1 for dot_ab, +2 for padding
    approx_col_total_data_width = max(
        len("Error"), # To fit "Error"
        max_approx_num_len + 1 + max_approx_den_len + 1 + # num\den and space
        max_approx_err_c_left + 1 + max_approx_err_c_right + # error_c parts and dot
        1 + # semicolon
        max_approx_err_ab_left + 1 + max_approx_err_ab_right + # error_ab parts and dot
        2 # padding
    )

    # Header will be "aXbY (x+Err_c);Err(X'Y)"
    max_header_approx_len = 0
    for b_val in current_b_chunk:
        max_header_approx_len = max(max_header_approx_len, len(f"a{current_a}b{b_val} (x+Err_c);Err({current_a}'{b_val})"))

    actual_approx_col_width = max(approx_col_total_data_width, max_header_approx_len)
    
    return col_widths, actual_approx_col_width, max_val_before_sep, max_val_after_sep, max_cents_left, max_cents_right, max_approx_num_len, max_approx_den_len, max_approx_err_c_left, max_approx_err_c_right, max_approx_err_ab_left, max_approx_err_ab_right

def print_tables(min_a, max_a, min_b, max_b, intervals_data, sort_order):
    """Processes and prints tables for each base 'a' with the specified sorting."""
    for a in range(min_a, max_a + 1):
        processed_intervals = []
        for original_str_repr, val_raw, monzo_raw, is_original_monzo_valid in intervals_data:
            
            # Use the updated normalize_interval which handles invalid monzos numerically
            normalized_monzo, normalized_val_actual, is_normalized_monzo_valid_for_display = normalize_interval(val_raw, monzo_raw, a, PRIMES, is_original_monzo_valid)
            
            # For display purposes, always use the numerically normalized value and its cents
            display_value_norm_raw = normalized_val_actual
            display_cents_norm = calculate_cents(normalized_val_actual)

            all_approximations_for_interval = []
            for b_val in range(min_b, max_b + 1):
                # Now, calculate approximation based on the numerically normalized value
                approx_fraction_str, error_c, error_ab = calculate_b_eda_approximation(display_value_norm_raw, a, b_val)
                all_approximations_for_interval.append((approx_fraction_str, error_c, error_ab))
            
            processed_intervals.append({
                'original_str_repr': original_str_repr,
                'monzo_normalized_raw': normalized_monzo,
                'monzo_display': format_monzo_vector_angle_brackets(normalized_monzo, is_normalized_monzo_valid_for_display),
                'value': normalized_val_actual, # This is the actual numerically normalized value
                'is_normalized_monzo_valid_for_display': is_normalized_monzo_valid_for_display, # Pass for formatting
                'cents': display_cents_norm, # This is the cents of the numerically normalized value
                'display_cents': display_cents_norm, # Store as number for potential later formatting
                'all_approximations': all_approximations_for_interval,
                'is_monzo_valid_for_sort': is_original_monzo_valid # Used for sorting
            })

        if sort_order == 'M':
            # Sort by monzo, putting "N.C." last. N.C. will have a monzo_display of "N.C." or an invalid monzo_normalized_raw.
            # We can use is_monzo_valid_for_sort and then the monzo values.
            processed_intervals.sort(key=lambda x: (not x['is_monzo_valid_for_sort'], [float(f) for f in x['monzo_normalized_raw']]))
        elif sort_order == 'C':
            processed_intervals.sort(key=lambda x: x['cents']) # Sort by actual numeric cents value
        else:
            print(f"Warning: Sorting option '{sort_order}' not recognized. No sorting applied.")

        b_values_range = list(range(min_b, max_b + 1))
        num_b_columns = len(b_values_range)
        
        # Max 6 approximations per table + 3 fixed columns
        APPROX_COLUMNS_PER_TABLE = 6 

        for chunk_start_idx in range(0, num_b_columns, APPROX_COLUMNS_PER_TABLE):
            current_b_chunk_values = b_values_range[chunk_start_idx : chunk_start_idx + APPROX_COLUMNS_PER_TABLE]

            col_widths, actual_approx_col_width, \
            max_val_before_sep, max_val_after_sep, \
            max_cents_left, max_cents_right, \
            max_approx_num_len, max_approx_den_len, \
            max_approx_err_c_left, max_approx_err_c_right, \
            max_approx_err_ab_left, max_approx_err_ab_right = \
                get_aligned_lengths(processed_intervals, current_b_chunk_values, min_b, max_b, a) # Pass 'a' here


            table_title = f"TABLE OF b-EDa APPROXIMATIONS for base 'a' = {a} (a.k.a. {a}-EDO period)"
            
            header_base_cols_names = ["Monzo Norm.", "Value", "Cents"]
            
            monzo_header_padded = f"{header_base_cols_names[0]:<{col_widths[header_base_cols_names[0]] + 2}}"
            valore_header_padded = f"{header_base_cols_names[1]:<{col_widths[header_base_cols_names[1]] + 2}}"
            cents_header_padded = f"{header_base_cols_names[2]:<{col_widths[header_base_cols_names[2]] + 2}}"

            approx_headers_with_err_info = []
            for b_val in current_b_chunk_values:
                # Update header for new error format
                approx_headers_with_err_info.append(f"a{a}b{b_val} (x+Err_c);Err({a}'{b_val})")
            
            # Construct header with all separators
            header_parts = [
                monzo_header_padded,
                valore_header_padded,
                cents_header_padded
            ]
            header_parts.extend([f"{h:<{actual_approx_col_width}}" for h in approx_headers_with_err_info])
            
            # Calculate total length for separator line and table borders
            total_header_length = sum([col_widths[name] + 2 for name in header_base_cols_names]) \
                                    + (actual_approx_col_width * len(current_b_chunk_values)) \
                                    + (len(header_base_cols_names) + len(current_b_chunk_values) - 1) * 3 # 3 is the length of " | "
            
            # Ensure the table borders are at least as long as the title
            border_length = max(total_header_length, len(table_title))

            print("\n" + "=" * border_length)
            print(table_title.center(border_length))
            print("=" * border_length)

            print(" | ".join(header_parts))
            print("-" * total_header_length) 

            for p_interval in processed_intervals:
                monzo_col = f"{p_interval['monzo_display']:<{col_widths['Monzo Norm.'] + 2}}"
                
                # Format value for alignment
                val_col = format_value_for_alignment(
                    p_interval['value'], 
                    is_monzo_rational(p_interval['monzo_normalized_raw']), 
                    p_interval['is_normalized_monzo_valid_for_display'],
                    max_val_before_sep, max_val_after_sep
                )
                # Apply extra padding for the column
                val_col_padded = f"{val_col:<{col_widths['Value'] + 2}}"

                # Ensure cents_str is numeric before formatting
                cents_str_val = p_interval['cents'] 
                cents_str_formatted = f"{cents_str_val:+.3f}" if isinstance(cents_str_val, (float, int)) else str(cents_str_val)
                
                if '.' in cents_str_formatted:
                    cl, cr = cents_str_formatted.split('.')
                    cents_col = f"{cl:>{max_cents_left}}.{cr:<{max_cents_right}}"
                else:
                    cents_col = f"{cents_str_formatted:>{max_cents_left}}"
                cents_col_padded = f"{cents_col:<{col_widths['Cents'] + 2}}"


                approximations_formatted_current_chunk = []
                for b_idx_in_full_list in range(len(p_interval['all_approximations'])):
                    current_b_from_idx = min_b + b_idx_in_full_list
                    if current_b_from_idx in current_b_chunk_values:
                        approx_fraction_str, error_c, error_ab = p_interval['all_approximations'][b_idx_in_full_list]
                        
                        if approx_fraction_str == "Invalid_Params":
                            full_approx_string = "Error"
                        else:
                            num_part, den_part = approx_fraction_str.split('\\')
                            error_c_str = f"{error_c:+.3f}" if isinstance(error_c, (float, int)) else str(error_c)
                            error_ab_str = f"{error_ab:+.3f}" if isinstance(error_ab, (float, int)) else str(error_ab)

                            formatted_fraction_part = f"{num_part:>{max_approx_num_len}}\\{den_part:<{max_approx_den_len}}"
                            
                            # Format error_c part
                            if '.' in error_c_str:
                                err_cl, err_cr = error_c_str.split('.')
                                formatted_error_c_part = f"{err_cl:>{max_approx_err_c_left}}.{err_cr:<{max_approx_err_c_right}}"
                            else:
                                formatted_error_c_part = f"{error_c_str:>{max_approx_err_c_left}}"

                            # Format error_ab part
                            if '.' in error_ab_str:
                                err_abl, err_abr = error_ab_str.split('.')
                                formatted_error_ab_part = f"{err_abl:>{max_approx_err_ab_left}}.{err_abr:<{max_approx_err_ab_right}}"
                            else:
                                formatted_error_ab_part = f"{error_ab_str:>{max_approx_err_ab_left}}"

                            full_approx_string = f"{formatted_fraction_part} {formatted_error_c_part};{formatted_error_ab_part}"
                        
                        approximations_formatted_current_chunk.append(f"{full_approx_string:<{actual_approx_col_width}}")

                # Construct data row with all separators
                data_parts = [
                    monzo_col,
                    val_col_padded,
                    cents_col_padded
                ]
                data_parts.extend(approximations_formatted_current_chunk)
                print(" | ".join(data_parts))

def get_intervals_from_user(current_input_mode_ref, intervals_data):
    """
    Function to handle user interval input and deletion.
    intervals_data is now passed by reference and modified directly.
    """
    
    print(f"\n--- Interval Input/Management (Current Mode: {current_input_mode_ref['mode']}) ('k' to finish, 'V' for Numerical Value, 'M' for Monzo, 'D' for Delete) ---")
    i = len(intervals_data) + 1 # Start counting from the next available interval

    while True:
        val = None
        monzo = None
        original_input_representation = ""
        is_completely_factorized = None
        
        if current_input_mode_ref['mode'] == 'M':
            val, monzo, new_mode_signal, is_completely_factorized = get_monzo_from_user_input(current_input_mode_ref, i)
            
            if new_mode_signal == 'k_terminate':
                break
            elif new_mode_signal == 'd_delete':
                if not intervals_data:
                    print("The interval list is already empty. No intervals to delete.")
                    continue
                # Call delete function
                delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
                i = len(intervals_data) + 1 # Update counter after deletion
                continue # Continue input/management loop
            elif new_mode_signal != current_input_mode_ref['mode']:
                continue
            
            if val is None and monzo is None: # If input was to change mode or invalid
                continue

            original_input_representation = format_monzo_vector_angle_brackets(monzo, is_completely_factorized)

        else: # current_input_mode_ref['mode'] == 'V'
            input_prompt = f"Interval {i} (Numerical Value): "
            input_received = input(input_prompt).strip()

            if input_received.lower() == 'k':
                break 
            elif input_received.upper() == 'V':
                if current_input_mode_ref['mode'] == 'V':
                    print("\nAlready in **Numerical Value** mode.")
                else:
                    current_input_mode_ref['mode'] = 'V'
                    print("\nMode changed to **Numerical Value**.")
                continue
            elif input_received.upper() == 'M':
                if current_input_mode_ref['mode'] == 'M':
                    print("\nAlready in **Fractional Monzo** mode.")
                else:
                    current_input_mode_ref['mode'] = 'M'
                    print("\nMode changed to **Fractional Monzo**.")
                continue
            elif input_received.upper() == 'D': # New 'D' option to delete
                if not intervals_data:
                    print("The interval list is already empty. No intervals to delete.")
                    continue
                delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
                i = len(intervals_data) + 1 # Update counter after deletion
                continue # Continue input/management loop

            val, monzo, is_completely_factorized = parse_value_input(input_received)
            original_input_representation = input_received
        
        if val is not None and monzo is not None and is_completely_factorized is not None:
            # Check for duplicates before adding
            is_duplicate = False
            for _, existing_val, existing_monzo, _ in intervals_data:
                # Compare by both value and monzo for robustness
                if abs(existing_val - val) < 1e-9 and existing_monzo == monzo:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print(f"The interval '{original_input_representation}' (Value: {val}) is already in the list. Please re-enter another interval.")
            else:
                intervals_data.append((original_input_representation, val, monzo, is_completely_factorized))
                i += 1
    # The function does not return anything, but modifies intervals_data directly.


def delete_interval(intervals_data, sort_order):
    """Allows the user to select and remove an interval from the list, sorting the display."""
    if not intervals_data:
        print("\nNo intervals present in the list to delete.")
        return

    print("\n--- Delete Interval ---")
    
    # Create a temporary list of dictionaries for easier sorting and display
    display_list = []
    # Use a set to track seen monzos in display_list to avoid duplicates in delete view
    seen_monzos_in_display = set() 

    for idx, (original_str_repr, val, monzo_raw, is_completely_factorized) in enumerate(intervals_data):
        monzo_tuple = tuple(monzo_raw)
        if monzo_tuple not in seen_monzos_in_display:
            display_list.append({
                'original_index': idx, # Store original index for actual deletion from intervals_data
                'original_str_repr': original_str_repr,
                'val': val, # This is the original input value (used for sorting N.C. and formatting)
                'monzo_raw': monzo_raw,
                'is_completely_factorized': is_completely_factorized,
            })
            seen_monzos_in_display.add(monzo_tuple)


    # Sort the display_list based on the current sort_order
    if sort_order == 'M':
        # Sort by factorization status (N.C. last), then by monzo
        display_list.sort(key=lambda x: (not x['is_completely_factorized'], [float(f) for f in x['monzo_raw']]))
        print("Current Intervals (sorted by lexicographical Monzo):")
    elif sort_order == 'C':
        # Sort by the 'val' (original value) which is always accurate
        display_list.sort(key=lambda x: calculate_cents(x['val'])) # Sort by cents of original value
        print("Current Intervals (sorted by ascending value):")
    else: # Default to C if sort_order is unknown
        display_list.sort(key=lambda x: calculate_cents(x['val'])) # Use original value for default sort
        print(f"Warning: Sorting option '{sort_order}' not recognized. Sorting by ascending value.")
    
    # Calculate max widths for consistent display
    max_idx_len = len(str(len(display_list)))
    max_orig_str_len = len("Original Interval")
    max_monzo_len = len("Relative Monzo")
    max_fact_len = len("Factorization")
    
    max_val_before_sep = 0
    max_val_after_sep = 0
    max_frac_str_len_val_col = 0 # Max length for the ;N/D part in Value column

    for item in display_list:
        max_orig_str_len = max(max_orig_str_len, len(item['original_str_repr']))
        max_monzo_len = max(max_monzo_len, len(format_monzo_vector_angle_brackets(item['monzo_raw'], item['is_completely_factorized'])))
        max_fact_len = max(max_fact_len, len(format_monzo_factorization(item['monzo_raw'], PRIMES, item['is_completely_factorized'])))
        
        # Calculate for value alignment in this display context
        val_for_display_calc = item['val'] # Always use the original val for length calculation here

        decimal_part_str_temp = f"{val_for_display_calc:.5f}"
        if '.' in decimal_part_str_temp:
            int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
            max_val_before_sep = max(max_val_before_sep, len(int_p_dec))
            max_val_after_sep = max(max_val_after_sep, len(frac_p_dec))
        else:
            max_val_before_sep = max(max_val_before_sep, len(decimal_part_str_temp))
            max_val_after_sep = max(max_val_after_sep, 0)
        
        if is_monzo_rational(item['monzo_raw']) and item['is_completely_factorized']:
            try:
                frac = Fraction(val_for_display_calc).limit_denominator(10**6)
                if frac.denominator != 1:
                    max_frac_str_len_val_col = max(max_frac_str_len_val_col, len(f";{frac.numerator}/{frac.denominator}"))
                elif frac.denominator == 1:
                    max_frac_str_len_val_col = max(max_frac_str_len_val_col, len(f";{frac.numerator}"))
            except (OverflowError, ZeroDivisionError, ValueError):
                pass
    
    # Value column total width for display list
    val_col_total_width = max(len("Value"), max_val_before_sep + (1 if max_val_after_sep > 0 else 0) + max_val_after_sep + max_frac_str_len_val_col)


    # Print header for deletion list
    header_parts = [
        f"{'#':<{max_idx_len}}",
        f"{'Original Interval':<{max_orig_str_len}}",
        f"{'Relative Monzo':<{max_monzo_len}}",
        f"{'Factorization':<{max_fact_len}}",
        f"{'Value':<{val_col_total_width}}"
    ]
    header_line = " | ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))

    for i, item in enumerate(display_list):
        # Crucial: Use item['val'] (original value) for formatting N.C. intervals
        val_to_format_actual = item['val'] # Always use original value from 'val' field for N.C. display

        formatted_value = format_value_for_alignment(
            val_to_format_actual, 
            is_monzo_rational(item['monzo_raw']), 
            item['is_completely_factorized'], 
            max_val_before_sep, max_val_after_sep
        )

        row_parts = [
            f"{i + 1:<{max_idx_len}}",
            f"{item['original_str_repr']:<{max_orig_str_len}}",
            f"{format_monzo_vector_angle_brackets(item['monzo_raw'], item['is_completely_factorized']):<{max_monzo_len}}",
            f"{format_monzo_factorization(item['monzo_raw'], PRIMES, item['is_completely_factorized']):<{max_fact_len}}",
            f"{formatted_value:<{val_col_total_width}}" # Ensure overall column width used here
        ]
        print(" | ".join(row_parts))

    while True:
        try:
            choice_str = input("Enter the number of the interval to delete (or '0' to cancel): ").strip()
            choice = int(choice_str)

            if choice == 0:
                print("Deletion canceled.")
                return
            
            if 1 <= choice <= len(display_list):
                # Get the original index from the sorted list to pop from the actual intervals_data
                original_index_to_delete = display_list[choice - 1]['original_index']
                removed_interval = intervals_data.pop(original_index_to_delete)
                print(f"Interval '{removed_interval[0]}' successfully removed.")
                break
            else:
                print("Invalid number. Please enter a number corresponding to an interval or '0' to cancel.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def is_monzo_rational(monzo_list):
    """Checks if all exponents in a Monzo are integers."""
    return all(exp.denominator == 1 for exp in monzo_list)


def print_summary_table(intervals_data, sort_order='None'):
    """Prints the summary of original intervals in tabular format."""
    
    unique_intervals_summary = []
    # Use a set to keep track of unique monzos (as tuples for hashability)
    seen_monzos = set() 

    # Collect data and track max lengths for alignment
    temp_max_val_before_sep = 0
    temp_max_val_after_sep = 0
    max_frac_str_len = 0 # Max length of the ";numerator/denominator" part

    for original_str_repr, val, monzo_raw, is_completely_factorized in intervals_data:
        monzo_tuple = tuple(monzo_raw)
        if monzo_tuple not in seen_monzos:
            calculated_value = calculate_monzo_value(monzo_raw, PRIMES)
            
            # --- Calculate max lengths for alignment for this table ---
            # Determine value to use for display length calculation
            val_for_display_calc = val if not is_completely_factorized else calculated_value

            # For the decimal part (before ';')
            decimal_part_str_temp = f"{val_for_display_calc:.5f}"
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(int_p_dec))
                temp_max_val_after_sep = max(temp_max_val_after_sep, len(frac_p_dec))
            else:
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(decimal_part_str_temp))
                temp_max_val_after_sep = max(temp_max_val_after_sep, 0)
            
            # For the fractional part (after ';')
            if is_monzo_rational(monzo_raw) and is_completely_factorized:
                try:
                    frac = Fraction(calculated_value).limit_denominator(10**6)
                    if frac.denominator != 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}/{frac.denominator}"))
                    elif frac.denominator == 1: # Integer case, still format as ';int'
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}"))
                except (OverflowError, ZeroDivisionError, ValueError):
                    pass # Will not have fraction part
            # --- End max length calculation ---

            unique_intervals_summary.append({
                'original_str_repr': original_str_repr,
                'monzo_raw': monzo_raw,
                'calculated_value': calculated_value,
                'val_original': val, # Store original value for N.C. display
                'is_completely_factorized': is_completely_factorized
            })
            seen_monzos.add(monzo_tuple)
    
    if sort_order == 'M':
        # Sort by monzo, putting "N.C." last
        unique_intervals_summary.sort(key=lambda x: (not x['is_completely_factorized'], [float(f) for f in x['monzo_raw']]))
    elif sort_order == 'C':
        # Sort by the original value for N.C. intervals, calculated value for factorizable ones
        unique_intervals_summary.sort(key=lambda x: calculate_cents(x['val_original']) if not x['is_completely_factorized'] else calculate_cents(x['calculated_value']))

    headers = ["Original Interval", "Relative Monzo", "Factorization", "Value"]
    
    # Calculate column widths based on data + headers
    col_widths = {header: len(header) for header in headers}
    
    table_rows_data_for_formatting = [] # Store raw data, then format for print
    for interval_info in unique_intervals_summary:
        original_str_repr = interval_info['original_str_repr']
        monzo_raw = interval_info['monzo_raw']
        calculated_value = interval_info['calculated_value']
        is_completely_factorized = interval_info['is_completely_factorized']
        val_original_for_nc_display = interval_info['val_original'] # For N.C. display

        formatted_monzo_vector = format_monzo_vector_angle_brackets(monzo_raw, is_completely_factorized)
        formatted_factorization = format_monzo_factorization(monzo_raw, PRIMES, is_completely_factorized)
        
        # Determine value to pass to format_value_for_alignment
        val_to_format_actual = val_original_for_nc_display if not is_completely_factorized else calculated_value

        # We don't format value here, it's done in the print loop with final alignment values
        table_rows_data_for_formatting.append({
            'original_str_repr': original_str_repr,
            'formatted_monzo_vector': formatted_monzo_vector,
            'formatted_factorization': formatted_factorization,
            'val_to_format_actual': val_to_format_actual, # This holds the float value
            'monzo_raw': monzo_raw, # Keep raw monzo for rationality check in format_value_for_alignment
            'is_completely_factorized': is_completely_factorized # Keep factorization status
        })

        # Update min width for other columns based on formatted data
        col_widths[headers[0]] = max(col_widths[headers[0]], len(original_str_repr))
        col_widths[headers[1]] = max(col_widths[headers[1]], len(formatted_monzo_vector))
        col_widths[headers[2]] = max(col_widths[headers[2]], len(formatted_factorization))
        # Value column width needs overall max length
        
    # Finalize Value column width: temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len
    # Plus potential padding
    col_widths[headers[3]] = max(len(headers[3]), temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len)


    # Print header
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    
    table_title = "TABLE OF ORIGINAL INTERVALS AND THEIR FACTORIZATION"
    # Ensure the table borders are at least as long as the title
    border_length = max(len(header_line), len(table_title))

    print("\n" + "=" * border_length)
    print(table_title.center(border_length))
    print("=" * border_length)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row_info in table_rows_data_for_formatting:
        val_col_str_aligned = format_value_for_alignment(
            row_info['val_to_format_actual'], 
            is_monzo_rational(row_info['monzo_raw']), 
            row_info['is_completely_factorized'], 
            temp_max_val_before_sep, temp_max_val_after_sep
        )
        row_line = " | ".join([f"{row_info['original_str_repr']:<{col_widths[headers[0]]}}",
                                f"{row_info['formatted_monzo_vector']:<{col_widths[headers[1]]}}",
                                f"{row_info['formatted_factorization']:<{col_widths[headers[2]]}}",
                                f"{val_col_str_aligned:<{col_widths[headers[3]]}}"]) # Ensure overall column width used here
        print(row_line)


def print_normalized_summary_table(normalization_base, intervals_data, sort_order='None'):
    """Prints the summary of normalized intervals in tabular format."""
    if normalization_base == 'w':
        print_summary_table(intervals_data, sort_order) # Revert to original summary table
        return

    unique_intervals_summary = []
    seen_monzos_original = set() 
    
    # Collect data and track max lengths for alignment
    temp_max_val_before_sep = 0
    temp_max_val_after_sep = 0
    max_frac_str_len = 0 # Max length of the ";numerator/denominator" part


    for original_str_repr, val_original, monzo_raw_original, is_original_monzo_valid in intervals_data:
        monzo_tuple_original = tuple(monzo_raw_original)
        if monzo_tuple_original not in seen_monzos_original:
            # Pass val_original to normalize_interval
            normalized_monzo, normalized_val_actual, is_normalized_monzo_valid_for_display = normalize_interval(val_original, monzo_raw_original, normalization_base, PRIMES, is_original_monzo_valid)
            
            # For display purposes, always use the numerically normalized value and its cents
            display_value_norm = normalized_val_actual
            display_cents_norm = calculate_cents(normalized_val_actual)

            # --- Calculate max lengths for alignment for this table ---
            is_rational_norm_monzo = is_monzo_rational(normalized_monzo)
            
            # For the decimal part (before ';')
            decimal_part_str_temp = f"{display_value_norm:.5f}" if isinstance(display_value_norm, (float, int, Fraction)) else str(display_value_norm)
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(int_p_dec))
                temp_max_val_after_sep = max(temp_max_val_after_sep, len(frac_p_dec))
            else:
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(decimal_part_str_temp))
                temp_max_val_after_sep = max(temp_max_val_after_sep, 0)
            
            # For the fractional part (after ';')
            if is_normalized_monzo_valid_for_display and is_rational_norm_monzo:
                try:
                    frac = Fraction(display_value_norm).limit_denominator(10**6)
                    if frac.denominator != 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}/{frac.denominator}"))
                    elif frac.denominator == 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}"))
                except (OverflowError, ZeroDivisionError, ValueError):
                    pass
            # --- End max length calculation ---

            unique_intervals_summary.append({
                'original_str_repr': original_str_repr,
                'normalized_monzo': normalized_monzo,
                'normalized_val': normalized_val_actual, # Value from normalized_interval
                'display_value': display_value_norm, # Value for display
                'normalized_cents': display_cents_norm, # Cents from normalized_interval
                'display_cents': f"{display_cents_norm:+.3f}", # Cents for display
                'is_normalized_monzo_valid': is_normalized_monzo_valid_for_display, # Validity of normalized monzo for formatting
                'is_original_monzo_valid': is_original_monzo_valid # Validity of original monzo (for sorting)
            })
            seen_monzos_original.add(monzo_tuple_original) # Add original monzo for uniqueness check

    if sort_order == 'M':
        # Sort by monzo, putting "N.C." last. We sort based on the original monzo validity
        unique_intervals_summary.sort(key=lambda x: (not x['is_original_monzo_valid'], [float(f) for f in x['normalized_monzo']]))
    elif sort_order == 'C':
        unique_intervals_summary.sort(key=lambda x: x['normalized_cents']) # Sort by actual numeric cents value
    else:
        print(f"Warning: Sorting option '{sort_order}' not recognized. No sorting applied.")

    headers = ["Original Interval", "Relative Monzo Norm.", "Factorization Norm.", "Value Norm.", "Cents Norm."]

    # Calculate column widths based on data + headers
    col_widths = {header: len(header) for header in headers}

    table_rows_data_for_formatting = []
    for interval_info in unique_intervals_summary:
        original_str_repr = interval_info['original_str_repr']
        normalized_monzo = interval_info['normalized_monzo']
        display_value = interval_info['display_value']
        display_cents = interval_info['display_cents']
        is_normalized_monzo_valid = interval_info['is_normalized_monzo_valid']

        formatted_factorization = format_monzo_factorization(normalized_monzo, PRIMES, is_normalized_monzo_valid)
        normalized_monzo_vector = format_monzo_vector_angle_brackets(normalized_monzo, is_normalized_monzo_valid)
        
        # Use the dedicated formatting function for alignment
        value_display_str_aligned = format_value_for_alignment(
            display_value, 
            is_monzo_rational(normalized_monzo), # Check rationality of normalized monzo
            is_normalized_monzo_valid, # Check factorization status of normalized monzo
            temp_max_val_before_sep, temp_max_val_after_sep
        )
        
        cents_display_str_padded = f"{display_cents:+.3f}" if isinstance(display_cents, (float, int)) else str(display_cents) # Defensive check

        table_rows_data_for_formatting.append([original_str_repr, normalized_monzo_vector, formatted_factorization, value_display_str_aligned, cents_display_str_padded])

        # Update min width for other columns based on formatted data
        col_widths[headers[0]] = max(col_widths[headers[0]], len(original_str_repr))
        col_widths[headers[1]] = max(col_widths[headers[1]], len(normalized_monzo_vector))
        col_widths[headers[2]] = max(col_widths[headers[2]], len(formatted_factorization))
        # Value Norm. width is set by temp_max_val_before_sep/after_sep logic
        col_widths[headers[4]] = max(col_widths[headers[4]], len(cents_display_str_padded))

    # Finalize Value Norm. column width: temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len
    col_widths[headers[3]] = max(len(headers[3]), temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len)

    # Print header
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    
    table_title = f"TABLE OF LOGARITHMIC NORMALIZED INTERVALS to base {normalization_base}"
    # Ensure the table borders are at least as long as the title
    border_length = max(len(header_line), len(table_title))

    print("\n" + "=" * border_length)
    print(table_title.center(border_length))
    print("=" * border_length)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row_info in table_rows_data_for_formatting:
        row_line = " | ".join([f"{row_info[0]:<{col_widths[headers[0]]}}",
                                f"{row_info[1]:<{col_widths[headers[1]]}}",
                                f"{row_info[2]:<{col_widths[headers[2]]}}",
                                f"{row_info[3]:<{col_widths[headers[3]]}}", # Already aligned by format_value_for_alignment
                                f"{row_info[4]:<{col_widths[headers[4]]}}"])
        print(row_line)

# --- MAIN PROGRAM ---

# Function to get valid r input, isolated for robustness
def get_valid_r_input():
    while True:
        a_input = input("\n--- Choose the normalization base r, with r>1 natural or r=w to return to the global view of original intervals --- \nr=").strip().lower()
        if a_input == 'w':
            return 'w'
        try:
            chosen_r_val = int(a_input)
            if chosen_r_val > 1:
                return chosen_r_val
            else:
                print("The base 'r' must be a natural number greater than 1.")
        except ValueError:
            print("Invalid input. Please enter a natural integer greater than 1 or 'w'.")


def main():
    # These will store the ranges for a and b
    global current_min_a, current_max_a, current_min_b, current_max_b, current_sort_order 

    print("--- b-EDa Approximations Calculator for Intervals ---")

    print("\n--- Interval Entry Mode Selection ---")
    print(" You can enter intervals in two ways:")
    print("   (V) for **Numerical Value**: enter integers (e.g., '5'), fractions (e.g., '3/2'), or decimal numbers (e.g., '2.46').")
    print("         Also, we support 'k^n' notation (e.g., '5^7') with k and n as integers.")
    print("         We also support simple expressions like '2*4*9'.")
    print("         Note: For radical intervals with fractional exponents (e.g., 2^(2/7), 5^(4/9)), please use Monzo with fractional values (e.g., [2/7⟩, [0 0 4/9⟩).")
    print("   (M) for **Fractional Monzo**: enter Monzo exponents one by one.")
    print("         Exponents can be integers (e.g., ‘-1’) or fractions (e.g., '1/2').")
    print("         This ensures maximum precision for fractional Monzos.")
    
    current_input_mode_ref = {'mode': input("Do you prefer to start by entering intervals (M) for Fractional Monzo or (V) for Numerical Value? [M/V]: ").strip().upper()}
    if current_input_mode_ref['mode'] not in ['V', 'M']:
        print("Invalid choice. Assuming Numerical Value (V) mode.")
        current_input_mode_ref['mode'] = 'V'

    intervals_data = [] # Each item will be (original_str_repr, val, monzo_raw, is_completely_factorized)
    
    # Interval entry
    get_intervals_from_user(current_input_mode_ref, intervals_data)

    if not intervals_data:
        print("No intervals entered. Exiting.")
        return

    # Prompt for temperament parameters after interval entry
    print("\n--- Enter Parameters for b-EDa Equal Temperaments ---")
    print("You can change these parameters later by typing 'T' in the final menu.")
    current_min_a = int(input("Enter the minimum value for base 'a': "))
    current_max_a = int(input("Enter the maximum value for base 'a': "))
    current_min_b = int(input("Enter the minimum value for 'b' (divisions): "))
    current_max_b = int(input("Enter the maximum value for 'b' (divisions): "))

    current_sort_order = input("\n--- Table Sorting ---\nHow do you want to sort intervals within each 'a'? Type 'M' for Monzo or 'C' for normalized Cents: ").strip().upper()
    if current_sort_order not in ['M', 'C']: 
        print("Invalid choice. Defaulting to Cents sorting.")
        current_sort_order = 'C'

    # Print tables
    print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
    
    # Print summary table
    print_summary_table(intervals_data, current_sort_order)


    # --- Loop to change sorting or enter/delete new intervals ---
    while True:
        print("\n--- Do you want to Reorder Tables, Manage Intervals, or Exit? ---")
        choice = input("Type 'T' to change b-EDa Equal Temperament parameters, 'R' to show summary modulo a certain parameter, 'M' to reorder by Monzo, 'C' to reorder by Cents, 'S' to enter more intervals, 'D' to delete an interval, or 'Q' to exit: ").strip().upper()

        if choice == 'Q':
            print("Exiting program.")
            break
        elif choice == 'S':
            get_intervals_from_user(current_input_mode_ref, intervals_data) 
            if not intervals_data: 
                print("No intervals present after modifications. Exiting.")
                break
            
            print("\nRegenerating tables with updated intervals...")
            print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
            print_summary_table(intervals_data, current_sort_order)


        elif choice == 'D': 
            if not intervals_data:
                print("The interval list is already empty. No intervals to delete.")
                continue
            delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
            if not intervals_data: 
                print("All intervals have been removed. Exiting program.")
                break
            
            print("\nRegenerating tables with updated intervals...")
            print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
            print_summary_table(intervals_data, current_sort_order)

        elif choice == 'R':
            # This is the line that calls the new, robust input function
            chosen_r_val = get_valid_r_input() 
            if chosen_r_val == 'w':
                print_summary_table(intervals_data, current_sort_order) # Use the new table function
            else: # It's a valid integer > 1
                print_normalized_summary_table(chosen_r_val, intervals_data, current_sort_order) # Use the new table function
            
        elif choice == 'T':
            print("\n--- Modify Parameters for b-EDa Equal Temperaments ---")
            try:
                new_min_a = int(input("Enter the NEW minimum value for base 'a': "))
                new_max_a = int(input("Enter the NEW maximum value for base 'a': "))
                new_min_b = int(input("Enter the NEW minimum value for 'b' (divisions): "))
                new_max_b = int(input("Enter the NEW maximum value for 'b' (divisions): "))

                current_min_a = new_min_a
                current_max_a = new_max_a
                current_min_b = new_min_b
                current_max_b = new_max_b

                print("\nTemperament parameters updated. Regenerating tables...")
                print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
                print_summary_table(intervals_data, current_sort_order) # Use the new table function
            except ValueError:
                print("Invalid input. Please ensure you enter integers for parameters.")

        elif choice in ['M', 'C']:
            if choice == current_sort_order:
                print(f"Tables are already sorted by {'Monzo' if current_sort_order == 'M' else 'Cents'}. Choose another option, 'S' to add intervals, 'D' to delete, 'T' to change parameters, 'R' for summary, or 'Q' to exit.")
            else:
                current_sort_order = choice
                print(f"\nReordering tables by {'Monzo' if current_sort_order == 'M' else 'Cents'}...")
                print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
                print_summary_table(intervals_data, current_sort_order) # Use the new table function
        else:
            print("Invalid choice. Type 'T', 'R', 'M', 'C', 'S', 'D' or 'Q'.")


if __name__ == "__main__":
    current_min_a = None
    current_max_a = None
    current_min_b = None
    current_max_b = None
    current_sort_order = None

    main()