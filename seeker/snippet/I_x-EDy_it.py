#date: 2025-07-09T16:58:46Z
#url: https://api.github.com/gists/064dd338edfe8b28d3550df147d00c03
#owner: https://api.github.com/users/gab-gash

import math
from fractions import Fraction

# --- FUNZIONI DI UTILITÀ PER MONZO E FATTORIZZAZIONE ---

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
]


def prime_factorize_fraction(number_fraction, primes_list):
    """
    Fattorizza una frazione (o un intero) nei suoi componenti primi.
    Restituisce un Monzo (lista di esponenti) e un booleano che indica se la fattorizzazione è completa.
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
    """Calcola i cents per un dato valore."""
    if value <= 0:
        return 0.0
    return 1200 * math.log2(float(value))

def normalize_interval(val_original, monzo, base_a, primes_list, is_monzo_valid):
    """
    Normalizza un intervallo rispetto a una base 'a' (il periodo),
    in modo che l'intervallo risultante sia >= 1 e < base_a.
    Questo elimina i cents negativi.

    Restituisce:
    - normalized_monzo: il monzo normalizzato (dummy se is_monzo_valid è False)
    - normalized_val: il valore numerico normalizzato (derivato dal monzo se valido, o da val_original se invalido)
    - is_normalized_monzo_valid: indica se il monzo normalizzato è significativo
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
    Calcola l'approssimazione b-EDa per un dato valore target,
    e restituisce la frazione x/b, l'errore in cents, e il nuovo errore Err(a'b).
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
    Forma una lista di Monzo (esponenti in Fraction) in una stringa
    leggibile che mostra la fattorizzazione in termini di potenze dei primi.
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
    Forma una lista di Monzo (esponenti in Fraction) in una stringa vettoriale
    compatta, troncando gli zeri finali per una migliore leggibilità, con parentesi angolate.
    """
    if not is_completely_factorized:
        return "N.C."

    last_nonzero_index = -1
    for i in range(len(monzo_list) - 1, -1, -1):
        if monzo_list[i] != Fraction(0, 1):
            last_nonzero_index = i
            break
    
    if last_nonzero_index == -1:
        return "[0⟩" # Formato [0⟩
    
    formatted_exponents = []
    for i in range(last_nonzero_index + 1):
        exponent = monzo_list[i]
        if exponent.denominator == 1:
            formatted_exponents.append(str(exponent.numerator))
        else:
            formatted_exponents.append(f"{exponent.numerator}/{exponent.denominator}")
    
    return f"[{' '.join(formatted_exponents)}⟩" # Formato [x y z⟩


def parse_value_input(input_str):
    """
    Analizza la stringa di input per un valore numerico (interi, frazioni, espressioni).
    Restituisce il valore numerico (float) e il Monzo.
    """
    try:
        local_dict = {'Fraction': Fraction, 'math': math}
        
        # Gestisce l'operatore di potenza '^' e lo rimpiazza con '**' di Python
        if '^' in input_str and '**' not in input_str:
            base_exp_parts = input_str.split('^')
            if len(base_exp_parts) == 2:
                base_str = base_exp_parts[0].strip()
                exp_str = base_exp_parts[1].strip()
                # Gestisce casi come 3^(1/4)
                if exp_str.startswith('(') and exp_str.endswith(')'):
                    exp_str = exp_str[1:-1] # Rimuove le parentesi per eval
                input_str = f"({base_str})**({exp_str})"

        val_eval = eval(input_str, {"__builtins__": None}, local_dict)

        if not isinstance(val_eval, (int, float, Fraction)):
            print(f"Errore: L'input '{input_str}' ha prodotto un tipo non supportato ({type(val_eval)}).")
            return None, None, None
            
        if float(val_eval) <= 0: # Ensure interval is positive
            print(f"Errore: Il valore dell'intervallo '{input_str}' deve essere strettamente positivo (> 0).")
            return None, None, None

        if isinstance(val_eval, int):
            fraction_val = Fraction(val_eval, 1)
        elif isinstance(val_eval, float):
            # Tenta di convertire il float in frazione, limitando il denominatore per evitare frazioni enormi da float imprecisi
            fraction_val = Fraction(val_eval).limit_denominator(10**6) 
        elif isinstance(val_eval, Fraction):
            fraction_val = val_eval

        monzo, is_completely_factorized = prime_factorize_fraction(fraction_val, PRIMES)
        
        return float(fraction_val), monzo, is_completely_factorized

    except (ValueError, SyntaxError, NameError, TypeError) as e:
        print(f"Errore di input per '{input_str}': {e}. Assicurati di usare un formato valido.")
        return None, None, None

def get_monzo_from_user_input(current_input_mode_ref, i):
    """
    Chiede all'utente di inserire gli esponenti di un Monzo uno alla volta.
    Restituisce un Monzo (lista di Fraction) e il suo valore numerico,
    o None, None se l'utente decide di finire/cambiare modalità.
    Ritorna anche la nuova modalità di input se cambiata.
    """
    monzo_exponents = []
    
    print("\n--- Inserimento Esponenti Monzo ---") # Mantiene il titolo per chiarezza ad ogni nuovo Monzo

    print("Digita 'f' per finire l'inserimento del Monzo.")
    print("In qualsiasi momento puoi digitare 'k' per annullare e terminare, 'V' per cambiare a Valore Numerico, 'D' per cancellare.") # Aggiornato prompt
    
    for idx, prime in enumerate(PRIMES):
        while True:
            exp_str = input(f"Esponente per ({prime}): ").strip().lower()
            
            if exp_str == 'f':
                # Riempi con zeri gli esponenti rimanenti
                monzo_exponents.extend([Fraction(0, 1)] * (len(PRIMES) - idx))
                # Stampa conferma Monzo
                monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
                # Ensure calculated value is positive (Monzo mode doesn't prevent negatives by input, only by resulting value)
                if monzo_val_calculated <= 0:
                    print(f"Errore: Il Monzo inserito risulta in un valore non positivo ({monzo_val_calculated}). Reinserisci.")
                    # Clear current exponents to restart for this monzo or allow user to exit
                    monzo_exponents.clear() 
                    continue # Re-ask for current prime exponent, effectively restarting this monzo input
                print(f"--- Intervallo {i} ha Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Formato [X Y Z⟩
                return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True
            
            elif exp_str == 'k':
                print("Inserimento Monzo annullato.")
                return None, None, 'k_terminate', None
            
            elif exp_str == 'v':
                current_input_mode_ref['mode'] = 'V'
                print("\nModalità cambiata a **Valore Numerico**.")
                print("Inserimento Monzo annullato.") # Se si cambia modalità, il Monzo corrente è annullato
                return None, None, current_input_mode_ref.get('mode'), None 
            
            elif exp_str == 'm': # Se l'utente digita 'm' mentre è già in modalità Monzo
                print("\nGià in modalità **Monzo Frazionario**.")
                print("Inserimento Monzo annullato.") # Annulla il Monzo corrente e riparte
                return None, None, current_input_mode_ref.get('mode'), None
            
            elif exp_str == 'd': # Opzione per cancellare durante l'inserimento Monzo
                return None, None, 'd_delete', None

            try:
                if '/' in exp_str:
                    num, den = map(int, exp_str.split('/'))
                    if den == 0:
                        raise ValueError("Denominatore non può essere zero.")
                    exponent = Fraction(num, den)
                else:
                    exponent = Fraction(int(exp_str), 1)
                monzo_exponents.append(exponent)
                break 
            except ValueError:
                print("Input non valido. Inserisci un numero intero o una frazione (es. '1/2').")
    
    # Se il ciclo finisce (tutti i PRIMES sono stati inseriti)
    monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
    if monzo_val_calculated <= 0:
        print(f"Errore: Il Monzo inserito risulta in un valore non positivo ({monzo_val_calculated}). Inserimento annullato.")
        return None, None, current_input_mode_ref.get('mode'), False # Treat as invalid
    print(f"--- Intervallo {i} ha Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Formato [X Y Z⟩
    return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True


def calculate_monzo_value(monzo_exponents, primes_list):
    """Calcola il valore numerico da un Monzo."""
    value = 1.0
    for i, exp in enumerate(monzo_exponents):
        if i < len(primes_list): 
            # Per Monzo razionale, esp può essere una Fraction intera o frazionaria
            # Se exp è una Fraction, float(exp) la converte in float, che gestisce anche le frazioni.
            value *= (primes_list[i] ** float(exp))
    return value


def format_value_for_alignment(value_to_format, is_rational_monzo, is_completely_factorized, total_width_before_decimal, total_width_after_decimal):
    """
    Formatta un valore numerico per l'allineamento.
    Genera una stringa nella forma "decimale;frazione" se applicabile, altrimenti solo "decimale".
    Allinea la parte decimale/frazionaria per i numeri.
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
    """Calcola le larghezze massime necessarie per allineare le colonne."""
    # Larghezze iniziali minime per le intestazioni
    col_widths = {
        "Monzo Norm.": len("Monzo Norm."), 
        "Valore": len("Valore"), 
        "Cents": len("Cents") 
    }
    
    max_val_before_sep = 0 # Max digits before '.' or '/'
    max_val_after_sep = 0  # Max digits after '.' or '/'
    max_val_full_string_len = len("Valore") # For overall column width of 'Valore' header

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
            # Correzione: Calcola l'indice 'b' corrente
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
    col_widths["Valore"] = max(max_val_full_string_len, len("Valore")) 

    # Aggiungi spazio extra per un buon padding e il punto decimale/segno
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
    """Processa e stampa le tabelle per ogni base 'a' con l'ordinamento specificato."""
    for a in range(min_a, max_a + 1):
        processed_intervals = []
        for original_str_repr, val_raw, monzo_raw, is_original_monzo_valid in intervals_data:
            
            # Use the updated normalize_interval which handles invalid monzos numerically
            normalized_monzo, normalized_val_actual, is_normalized_monzo_valid_for_display = normalize_interval(val_raw, monzo_raw, a, PRIMES, is_original_monzo_valid)
            
            # For display purposes, always use the numerically normalizzed value and its cents
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
            print(f"Avviso: Opzione di ordinamento '{sort_order}' non riconosciuta. Nessun ordinamento applicato.")

        b_values_range = list(range(min_b, max_b + 1))
        num_b_columns = len(b_values_range)
        
        # Max 6 approssimazioni per tabella + 3 colonne fisse
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


            table_title = f"TABELLA DELLE APPROSSIMAZIONI b-EDa per base 'a' = {a} (a.k.a. {a}-EDO period)"
            
            header_base_cols_names = ["Monzo Norm.", "Valore", "Cents"]
            
            monzo_header_padded = f"{header_base_cols_names[0]:<{col_widths[header_base_cols_names[0]] + 2}}"
            valore_header_padded = f"{header_base_cols_names[1]:<{col_widths[header_base_cols_names[1]] + 2}}"
            cents_header_padded = f"{header_base_cols_names[2]:<{col_widths[header_base_cols_names[2]] + 2}}"

            approx_headers_with_err_info = []
            for b_val in current_b_chunk_values:
                # Update header for new error format
                approx_headers_with_err_info.append(f"a{a}b{b_val} (x+Err_c);Err({a}'{b_val})")
            
            # Costruzione dell'intestazione con tutti i separatori
            header_parts = [
                monzo_header_padded,
                valore_header_padded,
                cents_header_padded
            ]
            header_parts.extend([f"{h:<{actual_approx_col_width}}" for h in approx_headers_with_err_info])
            
            # Calcola la lunghezza totale per la linea di separazione e i bordi della tabella
            total_header_length = sum([col_widths[name] + 2 for name in header_base_cols_names]) \
                                    + (actual_approx_col_width * len(current_b_chunk_values)) \
                                    + (len(header_base_cols_names) + len(current_b_chunk_values) - 1) * 3 # 3 è la lunghezza di " | "
            
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
                val_col_padded = f"{val_col:<{col_widths['Valore'] + 2}}"

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

                # Costruzione della riga di dati con tutti i separatori
                data_parts = [
                    monzo_col,
                    val_col_padded,
                    cents_col_padded
                ]
                data_parts.extend(approximations_formatted_current_chunk)
                print(" | ".join(data_parts))

def get_intervals_from_user(current_input_mode_ref, intervals_data):
    """
    Funzione per gestire l'inserimento e la cancellazione degli intervalli da parte dell'utente.
    intervals_data è ora passata per riferimento e modificata direttamente.
    """
    
    print(f"\n--- Inserimento/Gestione Intervalli (Modalità corrente: {current_input_mode_ref['mode']}) ('k' per finire, 'V' per Valore, 'M' per Monzo, 'D' per Cancellare) ---")
    i = len(intervals_data) + 1 # Inizia a contare dal prossimo intervallo disponibile

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
                    print("La lista degli intervalli è già vuota. Nessun intervallo da cancellare.")
                    continue
                # Chiamata alla funzione di cancellazione
                delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
                i = len(intervals_data) + 1 # Aggiorna il contatore dopo la cancellazione
                continue # Continua il loop di inserimento/gestione
            elif new_mode_signal != current_input_mode_ref['mode']:
                continue
            
            if val is None and monzo is None: # Se l'input era per cambiare modalità o non valido
                continue

            original_input_representation = format_monzo_vector_angle_brackets(monzo, is_completely_factorized)

        else: # current_input_mode_ref['mode'] == 'V'
            input_prompt = f"Intervallo {i} (Valore Numerico): "
            input_received = input(input_prompt).strip()

            if input_received.lower() == 'k':
                break 
            elif input_received.upper() == 'V':
                if current_input_mode_ref['mode'] == 'V':
                    print("\nGià in modalità **Valore Numerico**.")
                else:
                    current_input_mode_ref['mode'] = 'V'
                    print("\nModalità cambiata a **Valore Numerico**.")
                continue
            elif input_received.upper() == 'M':
                if current_input_mode_ref['mode'] == 'M':
                    print("\nGià in modalità **Monzo Frazionario**.")
                else:
                    current_input_mode_ref['mode'] = 'M'
                    print("\nModalità cambiata a **Monzo Frazionario**.")
                continue
            elif input_received.upper() == 'D': # Nuova opzione 'D' per cancellare
                if not intervals_data:
                    print("La lista degli intervalli è già vuota. Nessun intervallo da cancellare.")
                    continue
                delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
                i = len(intervals_data) + 1 # Aggiorna il contatore dopo la cancellazione
                continue # Continua il loop di inserimento/gestione

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
                print(f"L'intervallo '{original_input_representation}' (Valore: {val}) è già presente nella lista. Reinserisci un altro intervallo.")
            else:
                intervals_data.append((original_input_representation, val, monzo, is_completely_factorized))
                i += 1
    # La funzione non restituisce nulla, ma modifica intervals_data direttamente.


def delete_interval(intervals_data, sort_order):
    """Permette all'utente di selezionare e rimuovere un intervallo dalla lista, ordinando la visualizzazione."""
    if not intervals_data:
        print("\nNessun intervallo presente nella lista da cancellare.")
        return

    print("\n--- Cancella Intervallo ---")
    
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
        print("Intervalli attuali (ordinati per Monzo lessicografico):")
    elif sort_order == 'C':
        # Sort by the 'val' (original value) which is always accurate
        display_list.sort(key=lambda x: x['val'])
        print("Intervalli attuali (ordinati per valore crescente):")
    else: # Default to C if sort_order is unknown
        display_list.sort(key=lambda x: x['val']) # Use original value for default sort
        print(f"Avviso: Opzione di ordinamento '{sort_order}' non riconosciuta. Ordinamento per valore crescente.")
    
    # Calculate max widths for consistent display
    max_idx_len = len(str(len(display_list)))
    max_orig_str_len = len("Intervallo originale")
    max_monzo_len = len("Monzo relativo")
    max_fact_len = len("Fattorizzazzione")
    
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
    val_col_total_width = max(len("Valore"), max_val_before_sep + (1 if max_val_after_sep > 0 else 0) + max_val_after_sep + max_frac_str_len_val_col)


    # Print header for deletion list
    header_parts = [
        f"{'#':<{max_idx_len}}",
        f"{'Intervallo originale':<{max_orig_str_len}}",
        f"{'Monzo relativo':<{max_monzo_len}}",
        f"{'Fattorizzazzione':<{max_fact_len}}",
        f"{'Valore':<{val_col_total_width}}"
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
            choice_str = input("Digita il numero dell'intervallo da cancellare (o '0' per annullare): ").strip()
            choice = int(choice_str)

            if choice == 0:
                print("Cancellazione annullata.")
                return
            
            if 1 <= choice <= len(display_list):
                # Get the original index from the sorted list to pop from the actual intervals_data
                original_index_to_delete = display_list[choice - 1]['original_index']
                removed_interval = intervals_data.pop(original_index_to_delete)
                print(f"Intervallo '{removed_interval[0]}' rimosso con successo.")
                break
            else:
                print("Numero non valido. Inserisci un numero corrispondente a un intervallo o '0' per annullare.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")

def is_monzo_rational(monzo_list):
    """Controlla se tutti gli esponenti in un Monzo sono numeri interi."""
    return all(exp.denominator == 1 for exp in monzo_list)


def print_summary_table(intervals_data, sort_order='None'):
    """Stampa il riepilogo degli intervalli originali in formato tabellare."""
    
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

    headers = ["Intervallo originale", "Monzo relativo", "Fattorizzazzione", "Valore"]
    
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
    
    table_title = "TABELLA Intervalli originali inseriti e loro fattorizzazione"
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
    """Stampa il riepilogo degli intervalli normalizzati in formato tabellare."""
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
            
            # For display purposes, always use the numerically normalizzed value and its cents
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
        print(f"Avviso: Opzione di ordinamento '{sort_order}' non riconosciuta. Nessun ordinamento applicato.")

    headers = ["Intervallo originale", "Monzo relativo Norm.", "Fattorizzazzione Norm.", "Valore Norm.", "Cents Norm."]

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
    
    table_title = f"TABELLA normalizzazione logaritmica in base {normalization_base} degli Intervalli originali"
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

# --- PROGRAMMA PRINCIPALE ---

# Function to get valid r input, isolated for robustness
def get_valid_r_input():
    while True:
        a_input = input("\n--- Scegli la base di normalizzazione r, con r>1 naturale oppure r=w per tornare alla visualizzazione globale degli intervalli originali inseriti --- \nr=").strip().lower()
        if a_input == 'w':
            return 'w'
        try:
            chosen_r_val = int(a_input)
            if chosen_r_val > 1:
                return chosen_r_val
            else:
                print("La base 'r' deve essere un numero naturale maggiore di 1.")
        except ValueError:
            print("Input non valido. Inserisci un numero naturale intero maggiore di 1 o 'w'.")


def main():
    # These will store the ranges for a and b
    global current_min_a, current_max_a, current_min_b, current_max_b, current_sort_order 

    print("--- Calcolatore di Approssimazioni b-EDa per Intervalli ---")

    print("\n--- Scelta della Modalità di Inserimento Intervalli ---")
    print(" Puoi inserire gli intervalli in due modi:")
    print("   (V) per **Valore Numerico**: inserisci numeri interi (es. '5'), frazioni (es. '3/2'), o numeri con la virgola (es. '2.46').")
    print("         Inoltre, supportiamo la notazione 'k^n’ (es. '5^7') con k e n interi.")
    print("         Supportiamo anche espressioni semplici come '2*4*9'.")
    print("         Nota: Per gli intervalli radicali con esponente frazionario (es. 2^(2/7), 5^(4/9)), usa anzi il Monzo con valori frazionari (es [2/7⟩, [0 0 4/9⟩)")
    print("   (M) per **Monzo Frazionario**: inserisci gli esponenti del Monzo uno alla volta.")
    print("         Gli esponenti possono essere interi (es. ‘-1’) o frazioni (es. '1/2').")
    print("         Questo garantisce la massima precisione per i Monzo frazionari.")
    
    current_input_mode_ref = {'mode': input("Preferisci iniziare inserendo gli intervalli (M) per Monzo Frazionario o (V) per Valore Numerico? [M/V]: ").strip().upper()}
    if current_input_mode_ref['mode'] not in ['V', 'M']:
        print("Scelta non valida. Assumendo modalità Valore Numerico (V).")
        current_input_mode_ref['mode'] = 'V'

    intervals_data = [] # Each item will be (original_str_repr, val, monzo_raw, is_completely_factorized)
    
    # Inserimento degli intervalli
    get_intervals_from_user(current_input_mode_ref, intervals_data)

    if not intervals_data:
        print("Nessun intervallo inserito. Uscita.")
        return

    # Chiede i parametri dei temperamenti dopo l'inserimento degli intervalli
    print("\n--- Inserimento Parametri per i Temperamenti Equabili b-EDa ---")
    print("Potrai cambiare questi parametri successivamente digitando 'T' nel menu finale.")
    current_min_a = int(input("Inserisci il valore minimo per la base 'a': "))
    current_max_a = int(input("Inserisci il valore massimo per la base 'a': "))
    current_min_b = int(input("Inserisci il valore minimo per 'b' (divisioni): "))
    current_max_b = int(input("Inserisci il valore massimo per 'b' (divisioni): "))

    current_sort_order = input("\n--- Ordinamento Tabella ---\nCome vuoi ordinare gli intervalli all'interno di ogni 'a'? Digita 'M' per Monzo o 'C' per Cents normalizzati: ").strip().upper()
    if current_sort_order not in ['M', 'C']: 
        print("Scelta non valida. Ordinamento predefinito per Cents.")
        current_sort_order = 'C'

    # Stampa delle tabelle
    print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
    
    # Stampa del riepilogo in tabella
    print_summary_table(intervals_data, current_sort_order)


    # --- Ciclo per cambiare l'ordinamento o inserire/cancellare nuovi intervalli ---
    while True:
        print("\n--- Vuoi Riordinare le Tabelle, Gestire gli Intervalli oppure Uscire? ---")
        choice = input("Digita 'T' per cambiare i parametri dei Temperamenti equabili b-EDa, 'R' per mostrare il riepilogo modulo un certo parametro, 'M' per riordinare per Monzo, 'C' per Cents, 'S' per inserire altri intervalli, 'D' per cancellare un intervallo o 'Q' per uscire: ").strip().upper()

        if choice == 'Q':
            print("Uscita dal programma.")
            break
        elif choice == 'S':
            get_intervals_from_user(current_input_mode_ref, intervals_data) 
            if not intervals_data: 
                print("Nessun intervallo presente dopo le modifiche. Uscita.")
                break
            
            print("\nRigenerazione delle tabelle con gli intervalli aggiornati...")
            print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
            print_summary_table(intervals_data, current_sort_order)


        elif choice == 'D': 
            if not intervals_data:
                print("La lista degli intervalli è già vuota. Nessun intervallo da cancellare.")
                continue
            delete_interval(intervals_data, current_sort_order) # Pass current_sort_order
            if not intervals_data: 
                print("Tutti gli intervalli sono stati rimossi. Uscita dal programma.")
                break
            
            print("\nRigenerazione delle tabelle con gli intervalli aggiornati...")
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
            print("\n--- Modifica Parametri per i Temperamenti Equabili b-EDa ---")
            try:
                new_min_a = int(input("Inserisci il NUOVO valore minimo per la base 'a': "))
                new_max_a = int(input("Inserisci il NUOVO valore massimo per la base 'a': "))
                new_min_b = int(input("Inserisci il NUOVO valore minimo per 'b' (divisioni): "))
                new_max_b = int(input("Inserisci il NUOVO valore massimo per 'b' (divisioni): "))

                current_min_a = new_min_a
                current_max_a = new_max_a
                current_min_b = new_min_b
                current_max_b = new_max_b

                print("\nParametri dei temperamenti aggiornati. Rigenerazione delle tabelle...")
                print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
                print_summary_table(intervals_data, current_sort_order) # Use the new table function
            except ValueError:
                print("Input non valido. Assicurati di inserire numeri interi per i parametri.")

        elif choice in ['M', 'C']:
            if choice == current_sort_order:
                print(f"Le tabelle sono già ordinate per {'Monzo' if current_sort_order == 'M' else 'Cents'}. Scegli un'altra opzione, 'S' per aggiungere intervalli, 'D' per cancellare, 'T' per cambiare i parametri, 'R' per il riepilogo o 'Q' per uscire.")
            else:
                current_sort_order = choice
                print(f"\nRiordinamento delle tabelle per {'Monzo' if current_sort_order == 'M' else 'Cents'}...")
                print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
                print_summary_table(intervals_data, current_sort_order) # Use the new table function
        else:
            print("Scelta non valida. Digita 'T', 'R', 'M', 'C', 'S', 'D' o 'Q'.")


if __name__ == "__main__":
    current_min_a = None
    current_max_a = None
    current_min_b = None
    current_max_b = None
    current_sort_order = None

    main()