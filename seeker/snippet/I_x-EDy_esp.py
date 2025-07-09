#date: 2025-07-09T17:05:41Z
#url: https://api.github.com/gists/8961e87838218a5b6a515b445b03207a
#owner: https://api.github.com/users/gab-gash

import math
from fractions import Fraction

# --- FUNCIONES DE UTILIDAD PARA MONZO Y FACTORIZACIÓN ---

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
]


def prime_factorize_fraction(number_fraction, primes_list):
    """
    Factoriza una fracción (o un entero) en sus componentes primos.
    Devuelve un Monzo (lista de exponentes) y un booleano que indica si la factorización es completa.
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
    """Calcula los cents para un valor dado."""
    if value <= 0:
        return 0.0
    return 1200 * math.log2(float(value))

def normalize_interval(val_original, monzo, base_a, primes_list, is_monzo_valid):
    """
    Normaliza un Monzo con respecto a una base 'a' (el período),
    de modo que el intervalo resultante sea >= 1 y < base_a.
    Esto elimina los cents negativos.

    Devuelve:
    - normalized_monzo: el monzo normalizado (dummy si is_monzo_valid es False)
    - normalized_val: el valor numérico normalizado (derivado del monzo si es válido, o de val_original si es inválido)
    - is_normalized_monzo_valid: indica si el monzo normalizado es significativo
    """

    if base_a <= 0: # Maneja base_a inválida
        return [Fraction(0,1)] * len(primes_list), val_original, False # No se puede normalizar, devuelve valor original

    base_a_cents = calculate_cents(base_a)
    if base_a_cents == 0 and base_a != 1: # Base es 0 o negativa pero no 1
        return [Fraction(0,1)] * len(primes_list), val_original, False # No se puede normalizar

    if not is_monzo_valid:
        # Si el monzo original no es válido (ej., número no límite primo),
        # aún normalizamos numéricamente el valor y calculamos los cents a partir de él.
        # El monzo en sí seguirá siendo un dummy y se marcará como inválido para la visualización.
        current_val_for_normalization = val_original
        interval_cents = calculate_cents(current_val_for_normalization)

        if base_a_cents == 0: # Base es 1
            normalized_val = 1.0
        else:
            num_periods_to_shift = math.floor(interval_cents / base_a_cents)
            normalized_val = val_original / (base_a ** num_periods_to_shift)
        
        return [Fraction(0,1)] * len(primes_list), normalized_val, False
    else:
        # Si el monzo original es válido, procede con la normalización basada en monzo
        val_from_monzo = 1.0
        for i, exp in enumerate(monzo):
            val_from_monzo *= (primes_list[i] ** float(exp))
        
        if val_from_monzo <= 0:
            return [Fraction(0,1)] * len(primes_list), 1.0, True # No debería ocurrir con monzos válidos usualmente

        interval_cents = calculate_cents(val_from_monzo)

        if base_a_cents == 0: # Base es 1
            normalized_monzo = [Fraction(0,1)] * len(primes_list)
            normalized_val = 1.0
        else:
            base_a_fraction = Fraction(base_a, 1)
            base_a_monzo, _ = prime_factorize_fraction(base_a_fraction, primes_list) 
            
            num_periods_to_shift = math.floor(interval_cents / base_a_cents)

            normalized_monzo = list(monzo) # Copia para no modificar el original
            for i in range(len(normalized_monzo)):
                normalized_monzo[i] -= base_a_monzo[i] * num_periods_to_shift

            normalized_val = 1.0
            for i, exp in enumerate(normalized_monzo):
                normalized_val *= (primes_list[i] ** float(exp))
        
        return normalized_monzo, normalized_val, True


def calculate_b_eda_approximation(target_value, base_a, b):
    """
    Calcula la aproximación b-EDa para un valor objetivo dado,
    y devuelve la fracción x/b, el error en cents, y el nuevo error Err(a'b).
    """
    if base_a <= 0 or b <= 0 or target_value <= 0: # Verificación defensiva
        return "Invalid_Params", 0.0, 0.0

    target_cents = calculate_cents(target_value) # Usa calculate_cents seguro

    cents_per_base_a = calculate_cents(float(base_a)) # Usa calculate_cents seguro
    if cents_per_base_a == 0: # Base es 1
        return f"0\\{b}", 0.0, 0.0

    cents_per_tick = cents_per_base_a / b

    num_ticks = round(target_cents / cents_per_tick)
    
    # Calcula el valor aproximado (p/X) a partir de num_ticks
    # El valor aproximado por b-EDa es base_a^(num_ticks / b)
    approx_value = base_a ** (num_ticks / b)

    error_cents = calculate_cents(approx_value) - target_cents # Err(c) = cents de aprox - cents de objetivo

    # Calcula Err(a'b) = (100 * b / log2(base_a)) * log2(approx_value / target_value)
    # Esto asegura que Err(a'b) tenga el mismo signo que error_cents
    if base_a == 1 or math.log2(base_a) == 0: # Evita división por cero, usa cents para consistencia
        new_error_ab = 0.0
    elif target_value <= 0 or approx_value <= 0: # Los argumentos logarítmicos deben ser positivos
        new_error_ab = 0.0
    else:
        new_error_ab = (100 * b / math.log2(base_a)) * math.log2(approx_value / target_value)

    return f"{int(num_ticks)}\\{b}", error_cents, new_error_ab


def format_monzo_factorization(monzo_list, primes, is_completely_factorized):
    """
    Formatea una lista Monzo (exponentes de Fracción) en una cadena legible
    que muestra la factorización prima.
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
    Formatea una lista Monzo (exponentes de Fracción) en una cadena de vector compacta,
    truncando ceros finales para legibilidad, con corchetes angulares.
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
    Analiza la cadena de entrada para un valor numérico (enteros, fracciones, expresiones).
    Devuelve el valor numérico (float) y el Monzo.
    """
    try:
        local_dict = {'Fraction': Fraction, 'math': math}
        
        # Maneja el operador de potencia '^' y lo reemplaza con el '**' de Python
        if '^' in input_str and '**' not in input_str:
            base_exp_parts = input_str.split('^')
            if len(base_exp_parts) == 2:
                base_str = base_exp_parts[0].strip()
                exp_str = base_exp_parts[1].strip()
                # Maneja casos como 3^(1/4)
                if exp_str.startswith('(') and exp_str.endswith(')'):
                    exp_str = exp_str[1:-1] # Elimina paréntesis para eval
                input_str = f"({base_str})**({exp_str})"

        val_eval = eval(input_str, {"__builtins__": None}, local_dict)

        if not isinstance(val_eval, (int, float, Fraction)):
            print(f"Error: La entrada '{input_str}' produjo un tipo no soportado ({type(val_eval)}).")
            return None, None, None
            
        if float(val_eval) <= 0: # Asegura que el intervalo sea positivo
            print(f"Error: El valor del intervalo '{input_str}' debe ser estrictamente positivo (> 0).")
            return None, None, None

        if isinstance(val_eval, int):
            fraction_val = Fraction(val_eval, 1)
        elif isinstance(val_eval, float):
            # Intenta convertir float a fracción, limitando el denominador para evitar fracciones enormes por floats imprecisos
            fraction_val = Fraction(val_eval).limit_denominator(10**6) 
        elif isinstance(val_eval, Fraction):
            fraction_val = val_eval

        monzo, is_completely_factorized = prime_factorize_fraction(fraction_val, PRIMES)
        
        return float(fraction_val), monzo, is_completely_factorized

    except (ValueError, SyntaxError, NameError, TypeError) as e:
        print(f"Error de entrada para '{input_str}': {e}. Asegúrese de usar un formato válido.")
        return None, None, None

def get_monzo_from_user_input(current_input_mode_ref, i):
    """
    Solicita al usuario que ingrese los exponentes del Monzo uno por uno.
    Devuelve un Monzo (lista de Fracciones) y su valor numérico,
    o None, None si el usuario decide finalizar/cambiar de modo.
    También devuelve el nuevo modo de entrada si se cambió.
    """
    monzo_exponents = []
    
    print("\n--- Entrada de Exponentes de Monzo ---") # Mantiene el título claro para cada nuevo Monzo

    print("Escriba 'f' para finalizar la entrada del Monzo.")
    print("En cualquier momento, puede escribir 'k' para cancelar y salir, 'V' para cambiar a Valor Numérico, 'D' para eliminar.") # Mensaje actualizado
    
    for idx, prime in enumerate(PRIMES):
        while True:
            exp_str = input(f"Exponente para ({prime}): ").strip().lower()
            
            if exp_str == 'f':
                # Rellena con ceros los exponentes restantes
                monzo_exponents.extend([Fraction(0, 1)] * (len(PRIMES) - idx))
                # Imprime confirmación del Monzo
                monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
                # Asegura que el valor calculado sea positivo (el modo Monzo no previene negativos por entrada, solo por el valor resultante)
                if monzo_val_calculated <= 0:
                    print(f"Error: El Monzo ingresado resulta en un valor no positivo ({monzo_val_calculated}). Por favor, ingrese de nuevo.")
                    # Limpia los exponentes actuales para reiniciar la entrada de este monzo o permite al usuario salir
                    monzo_exponents.clear() 
                    continue # Vuelve a preguntar por el exponente primo actual, reiniciando efectivamente esta entrada de monzo
                print(f"--- El Intervalo {i} tiene Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Formato [X Y Z⟩
                return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True
            
            elif exp_str == 'k':
                print("Entrada de Monzo cancelada.")
                return None, None, 'k_terminate', None
            
            elif exp_str == 'v':
                current_input_mode_ref['mode'] = 'V'
                print("\nModo cambiado a **Valor Numérico**.")
                print("Entrada de Monzo cancelada.") # Si se cambia el modo, el Monzo actual se cancela
                return None, None, current_input_mode_ref.get('mode'), None 
            
            elif exp_str == 'm': # Si el usuario escribe 'm' mientras ya está en modo Monzo
                print("\nYa está en modo **Monzo Fraccionario**.")
                print("Entrada de Monzo cancelada.") # Cancela el Monzo actual y reinicia
                return None, None, current_input_mode_ref.get('mode'), None
            
            elif exp_str == 'd': # Opción para eliminar durante la entrada del Monzo
                return None, None, 'd_delete', None

            try:
                if '/' in exp_str:
                    num, den = map(int, exp_str.split('/'))
                    if den == 0:
                        raise ValueError("El denominador no puede ser cero.")
                    exponent = Fraction(num, den)
                else:
                    exponent = Fraction(int(exp_str), 1)
                monzo_exponents.append(exponent)
                break 
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número entero o una fracción (ej., '1/2').")
    
    # Si el bucle termina (se han ingresado todos los PRIMES)
    monzo_val_calculated = calculate_monzo_value(monzo_exponents, PRIMES)
    if monzo_val_calculated <= 0:
        print(f"Error: El Monzo ingresado resulta en un valor no positivo ({monzo_val_calculated}). Entrada cancelada.")
        return None, None, current_input_mode_ref.get('mode'), False # Tratar como inválido
    print(f"--- El Intervalo {i} tiene Monzo: {format_monzo_vector_angle_brackets(monzo_exponents, True)} ---") # Formato [X Y Z⟩
    return monzo_val_calculated, monzo_exponents, current_input_mode_ref.get('mode'), True


def calculate_monzo_value(monzo_exponents, primes_list):
    """Calcula el valor numérico a partir de un Monzo."""
    value = 1.0
    for i, exp in enumerate(monzo_exponents):
        if i < len(primes_list): 
            # Para Monzo racional, exp puede ser una Fracción entera o fraccionaria
            # Si exp es una Fracción, float(exp) lo convierte a float, que también maneja fracciones.
            value *= (primes_list[i] ** float(exp))
    return value


def format_value_for_alignment(value_to_format, is_rational_monzo, is_completely_factorized, total_width_before_decimal, total_width_after_decimal):
    """
    Formatea un valor numérico para alineación.
    Genera una cadena en la forma "decimal;fracción" si aplica, de lo contrario solo "decimal".
    Alinea la parte decimal/fraccionaria para los números.
    """
    # Asegura que value_to_format sea numérico para el formato
    if not isinstance(value_to_format, (int, float, Fraction)):
        return str(value_to_format) # Devuelve tal cual si no es numérico

    decimal_part_str = f"{float(value_to_format):.5f}" # Siempre convierte a float para un formato decimal consistente
    
    fraction_part_str_formatted = ""
    # Solo intenta mostrar la fracción si es racional y está completamente factorizada
    if is_rational_monzo and is_completely_factorized:
        try:
            frac = Fraction(value_to_format).limit_denominator(10**6)
            # Solo muestra la fracción si no es un entero exacto (ej., 2/1) o si el usuario quiere mostrar explícitamente el entero como fracción
            if frac.denominator != 1:
                fraction_part_str_formatted = f";{frac.numerator}/{frac.denominator}"
            elif frac.denominator == 1: # Para enteros, mostrar como ;N si se solicita
                fraction_part_str_formatted = f";{frac.numerator}" # El usuario quiere el entero como fracción también
        except (OverflowError, ZeroDivisionError, ValueError):
            pass # Si la conversión a Fracción falla, usa solo el decimal

    # Divide la parte decimal en componentes enteros y fraccionarios para alineación
    if '.' in decimal_part_str:
        int_part_dec, frac_part_dec = decimal_part_str.split('.')
        separator = "."
    else: # Es un entero o float sin parte decimal explícita
        int_part_dec = decimal_part_str
        frac_part_dec = "" 
        separator = ""
    
    # Rellena la parte entera para alineación
    padded_int_part = f"{int_part_dec:>{total_width_before_decimal}}"
    
    # Rellena la parte fraccionaria para alineación
    # Si no hay parte fraccionaria, rellena con espacios para alinear con otros que sí la tengan
    padded_frac_part = f"{frac_part_dec:<{total_width_after_decimal}}"

    # Combina las partes
    if frac_part_dec or total_width_after_decimal > 0: # Si hay una parte decimal o necesitamos reservar espacio para ella
        aligned_decimal_str = f"{padded_int_part}{separator}{padded_frac_part}"
    else: # Entero puro, sin punto decimal, solo rellena la parte entera
        aligned_decimal_str = padded_int_part

    return f"{aligned_decimal_str}{fraction_part_str_formatted}"


def get_aligned_lengths(processed_intervals, current_b_chunk, min_b, max_b, current_a):
    """Calcula los anchos máximos requeridos para alinear columnas."""
    # Anchos iniciales mínimos para los encabezados
    col_widths = {
        "Monzo Norm.": len("Monzo Norm."), 
        "Valor": len("Valor"), 
        "Cents": len("Cents") 
    }
    
    max_val_before_sep = 0 # Máximo de dígitos antes de '.' o '/'
    max_val_after_sep = 0  # Máximo de dígitos después de '.' o '/'
    max_val_full_string_len = len("Valor") # Para el ancho total de la columna del encabezado 'Valor'

    max_cents_left = 0
    max_cents_right = 0
    
    # Nuevas variables para error_c y error_ab
    max_approx_num_len = 0 
    max_approx_den_len = 0 
    max_approx_err_c_left = 0
    max_approx_err_c_right = 0
    max_approx_err_ab_left = 0
    max_approx_err_ab_right = 0 

    for p_interval in processed_intervals:
        col_widths["Monzo Norm."] = max(col_widths["Monzo Norm."], len(p_interval['monzo_display']))

        # Calcula los anchos máximos para la alineación de la columna de Valor
        val_to_check = p_interval['value'] # Usa el valor numérico real para determinar el formato
        is_rational_to_format = is_monzo_rational(p_interval['monzo_normalized_raw']) and p_interval['is_normalized_monzo_valid_for_display']

        # Determina la longitud de la parte decimal
        # Asegura que val_to_check sea numérico antes de formatear
        if isinstance(val_to_check, (float, int, Fraction)):
            decimal_part_str_temp = f"{float(val_to_check):.5f}"
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                max_val_before_sep = max(max_val_before_sep, len(int_p_dec))
                max_val_after_sep = max(max_val_after_sep, len(frac_p_dec))
            else:
                max_val_before_sep = max(max_val_before_sep, len(decimal_part_str_temp))
                max_val_after_sep = max(max_val_after_sep, 0)
        else: # Maneja valores no numéricos como "Error" para el cálculo del ancho
            max_val_before_sep = max(max_val_before_sep, len(str(val_to_check)))
            max_val_after_sep = max(max_val_after_sep, 0)

        # Determina la longitud completa de la cadena formateada, incluyendo la fracción opcional para el ancho total de la columna
        temp_formatted_value_string = format_value_for_alignment(
            val_to_check, 
            is_rational_to_format, 
            p_interval['is_normalized_monzo_valid_for_display'], # Usa el estado de factorización real para la parte de la fracción
            max_val_before_sep, # Usa max_val_before_sep preliminar
            max_val_after_sep # Usa max_val_after_sep preliminar
        )
        max_val_full_string_len = max(max_val_full_string_len, len(temp_formatted_value_string))


        cents_str_val = p_interval['display_cents'] # Usa display_cents para el cálculo del ancho
        cents_str_formatted = f"{cents_str_val:+.3f}" if isinstance(cents_str_val, (float, int)) else str(cents_str_val) # Verificación defensiva

        if '.' in cents_str_formatted:
            cl, cr = cents_str_formatted.split('.')
            max_cents_left = max(max_cents_left, len(cl))
            max_cents_right = max(max_cents_right, len(cr))
        else:
            max_cents_left = max(max_cents_left, len(cents_str_formatted))

        for b_val_actual_idx in range(len(p_interval['all_approximations'])):
            # Corrección: Calcula el índice 'b' actual
            current_b_from_idx = min_b + b_val_actual_idx
            if current_b_from_idx in current_b_chunk: # Solo para columnas en el fragmento actual
                approx_fraction_str, error_c, error_ab = p_interval['all_approximations'][b_val_actual_idx]
                if approx_fraction_str != "Invalid_Params":
                    num_part, den_part = approx_fraction_str.split('\\')
                    
                    max_approx_num_len = max(max_approx_num_len, len(num_part))
                    max_approx_den_len = max(max_approx_den_len, len(den_part))

                    error_c_str = f"{error_c:+.3f}" if isinstance(error_c, (float, int)) else str(error_c) # Verificación defensiva
                    if '.' in error_c_str:
                        err_cl, err_cr = error_c_str.split('.')
                        max_approx_err_c_left = max(max_approx_err_c_left, len(err_cl))
                        max_approx_err_c_right = max(max_approx_err_c_right, len(err_cr))
                    else:
                        max_approx_err_c_left = max(max_approx_err_c_left, len(error_c_str))

                    error_ab_str = f"{error_ab:+.3f}" if isinstance(error_ab, (float, int)) else str(error_ab) # Verificación defensiva
                    if '.' in error_ab_str:
                        err_abl, err_abr = error_ab_str.split('.')
                        max_approx_err_ab_left = max(max_approx_err_ab_left, len(err_abl))
                        max_approx_err_ab_right = max(max_approx_err_ab_right, len(err_abr))
                    else:
                        max_approx_err_ab_left = max(max_approx_err_ab_left, len(error_ab_str))
                else: # Maneja "Invalid_Params" para el cálculo del ancho
                    max_approx_num_len = max(max_approx_num_len, len("Error")) # Para la cadena "Error"
                    max_approx_den_len = max(max_approx_den_len, 0)
                    max_approx_err_c_left = max(max_approx_err_c_left, 0)
                    max_approx_err_c_right = max(max_approx_err_c_right, 0)
                    max_approx_err_ab_left = max(max_approx_err_ab_left, 0)
                    max_approx_err_ab_right = max(max_approx_err_ab_right, 0)


    # Finaliza el ancho de la columna de Valor basándose en la alineación del separador Y la longitud completa de la cadena
    col_widths["Value"] = max(max_val_full_string_len, len("Value")) 

    # Agrega espacio extra para un buen relleno y el punto decimal/signo
    col_widths["Cents"] = max(col_widths["Cents"], max_cents_left + 1 + max_cents_right)

    # Nuevo cálculo del ancho total de la columna de aproximación:
    # num\den + espacio + signo_c_digitos_c.decimales_c + punto y coma + signo_ab_digitos_ab.decimales_ab + relleno
    # +1 para '\', +1 para espacio, +1 para punto_c, +1 para punto y coma, +1 para punto_ab, +2 para relleno
    approx_col_total_data_width = max(
        len("Error"), # Para que quepa "Error"
        max_approx_num_len + 1 + max_approx_den_len + 1 + # num\den y espacio
        max_approx_err_c_left + 1 + max_approx_err_c_right + # partes de error_c y punto
        1 + # punto y coma
        max_approx_err_ab_left + 1 + max_approx_err_ab_right + # partes de error_ab y punto
        2 # relleno
    )

    # El encabezado será "aXbY (x+Err_c);Err(X'Y)"
    max_header_approx_len = 0
    for b_val in current_b_chunk:
        max_header_approx_len = max(max_header_approx_len, len(f"a{current_a}b{b_val} (x+Err_c);Err({current_a}'{b_val})"))

    actual_approx_col_width = max(approx_col_total_data_width, max_header_approx_len)
    
    return col_widths, actual_approx_col_width, max_val_before_sep, max_val_after_sep, max_cents_left, max_cents_right, max_approx_num_len, max_approx_den_len, max_approx_err_c_left, max_approx_err_c_right, max_approx_err_ab_left, max_approx_err_ab_right

def print_tables(min_a, max_a, min_b, max_b, intervals_data, sort_order):
    """Procesa e imprime tablas para cada base 'a' con la clasificación especificada."""
    for a in range(min_a, max_a + 1):
        processed_intervals = []
        for original_str_repr, val_raw, monzo_raw, is_original_monzo_valid in intervals_data:
            
            # Usa la función normalize_interval actualizada que maneja numéricamente los monzos inválidos
            normalized_monzo, normalized_val_actual, is_normalized_monzo_valid_for_display = normalize_interval(val_raw, monzo_raw, a, PRIMES, is_original_monzo_valid)
            
            # Para fines de visualización, siempre usa el valor numéricamente normalizado y sus cents
            display_value_norm_raw = normalized_val_actual
            display_cents_norm = calculate_cents(normalized_val_actual)

            all_approximations_for_interval = []
            for b_val in range(min_b, max_b + 1):
                # Ahora, calcula la aproximación basándose en el valor numéricamente normalizado
                approx_fraction_str, error_c, error_ab = calculate_b_eda_approximation(display_value_norm_raw, a, b_val)
                all_approximations_for_interval.append((approx_fraction_str, error_c, error_ab))
            
            processed_intervals.append({
                'original_str_repr': original_str_repr,
                'monzo_normalized_raw': normalized_monzo,
                'monzo_display': format_monzo_vector_angle_brackets(normalized_monzo, is_normalized_monzo_valid_for_display),
                'value': normalized_val_actual, # Este es el valor numéricamente normalizado real
                'is_normalized_monzo_valid_for_display': is_normalized_monzo_valid_for_display, # Pasa para el formato
                'cents': display_cents_norm, # Estos son los cents del valor numéricamente normalizado
                'display_cents': display_cents_norm, # Almacena como número para un posible formato posterior
                'all_approximations': all_approximations_for_interval,
                'is_monzo_valid_for_sort': is_original_monzo_valid # Usado para la clasificación
            })

        if sort_order == 'M':
            # Clasifica por monzo, poniendo "N.C." al final. "N.C." tendrá un monzo_display de "N.C." o un monzo_normalized_raw inválido.
            # Podemos usar is_monzo_valid_for_sort y luego los valores de monzo.
            processed_intervals.sort(key=lambda x: (not x['is_monzo_valid_for_sort'], [float(f) for f in x['monzo_normalized_raw']]))
        elif sort_order == 'C':
            processed_intervals.sort(key=lambda x: x['cents']) # Clasifica por el valor numérico real de los cents
        else:
            print(f"Advertencia: La opción de clasificación '{sort_order}' no reconocida. No se aplicó ninguna clasificación.")

        b_values_range = list(range(min_b, max_b + 1))
        num_b_columns = len(b_values_range)
        
        # Máximo de 6 aproximaciones por tabla + 3 columnas fijas
        APPROX_COLUMNS_PER_TABLE = 6 

        for chunk_start_idx in range(0, num_b_columns, APPROX_COLUMNS_PER_TABLE):
            current_b_chunk_values = b_values_range[chunk_start_idx : chunk_start_idx + APPROX_COLUMNS_PER_TABLE]

            col_widths, actual_approx_col_width, \
            max_val_before_sep, max_val_after_sep, \
            max_cents_left, max_cents_right, \
            max_approx_num_len, max_approx_den_len, \
            max_approx_err_c_left, max_approx_err_c_right, \
            max_approx_err_ab_left, max_approx_err_ab_right = \
                get_aligned_lengths(processed_intervals, current_b_chunk_values, min_b, max_b, a) # Pasa 'a' aquí


            table_title = f"TABLA DE APROXIMACIONES b-EDa para la base 'a' = {a} (también conocido como {a}-EDO período)"
            
            header_base_cols_names = ["Monzo Norm.", "Valor", "Cents"]
            
            monzo_header_padded = f"{header_base_cols_names[0]:<{col_widths[header_base_cols_names[0]] + 2}}"
            valore_header_padded = f"{header_base_cols_names[1]:<{col_widths[header_base_cols_names[1]] + 2}}"
            cents_header_padded = f"{header_base_cols_names[2]:<{col_widths[header_base_cols_names[2]] + 2}}"

            approx_headers_with_err_info = []
            for b_val in current_b_chunk_values:
                # Actualiza el encabezado para el nuevo formato de error
                approx_headers_with_err_info.append(f"a{a}b{b_val} (x+Err_c);Err({a}'{b_val})")
            
            # Construye el encabezado con todos los separadores
            header_parts = [
                monzo_header_padded,
                valore_header_padded,
                cents_header_padded
            ]
            header_parts.extend([f"{h:<{actual_approx_col_width}}" for h in approx_headers_with_err_info])
            
            # Calcula la longitud total de la línea separadora y los bordes de la tabla
            total_header_length = sum([col_widths[name] + 2 for name in header_base_cols_names]) \
                                    + (actual_approx_col_width * len(current_b_chunk_values)) \
                                    + (len(header_base_cols_names) + len(current_b_chunk_values) - 1) * 3 # 3 es la longitud de " | "
            
            # Asegura que los bordes de la tabla sean al menos tan largos como el título
            border_length = max(total_header_length, len(table_title))

            print("\n" + "=" * border_length)
            print(table_title.center(border_length))
            print("=" * border_length)

            print(" | ".join(header_parts))
            print("-" * total_header_length) 

            for p_interval in processed_intervals:
                monzo_col = f"{p_interval['monzo_display']:<{col_widths['Monzo Norm.'] + 2}}"
                
                # Formato de valor para alineación
                val_col = format_value_for_alignment(
                    p_interval['value'], 
                    is_monzo_rational(p_interval['monzo_normalized_raw']), 
                    p_interval['is_normalized_monzo_valid_for_display'],
                    max_val_before_sep, max_val_after_sep
                )
                # Aplica relleno extra para la columna
                val_col_padded = f"{val_col:<{col_widths['Valor'] + 2}}"

                # Asegura que cents_str sea numérico antes de formatear
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
                            
                            # Formato de la parte error_c
                            if '.' in error_c_str:
                                err_cl, err_cr = error_c_str.split('.')
                                formatted_error_c_part = f"{err_cl:>{max_approx_err_c_left}}.{err_cr:<{max_approx_err_c_right}}"
                            else:
                                formatted_error_c_part = f"{error_c_str:>{max_approx_err_c_left}}"

                            # Formato de la parte error_ab
                            if '.' in error_ab_str:
                                err_abl, err_abr = error_ab_str.split('.')
                                formatted_error_ab_part = f"{err_abl:>{max_approx_err_ab_left}}.{err_abr:<{max_approx_err_ab_right}}"
                            else:
                                formatted_error_ab_part = f"{error_ab_str:>{max_approx_err_ab_left}}"

                            full_approx_string = f"{formatted_fraction_part} {formatted_error_c_part};{formatted_error_ab_part}"
                        
                        approximations_formatted_current_chunk.append(f"{full_approx_string:<{actual_approx_col_width}}")

                # Construye la fila de datos con todos los separadores
                data_parts = [
                    monzo_col,
                    val_col_padded,
                    cents_col_padded
                ]
                data_parts.extend(approximations_formatted_current_chunk)
                print(" | ".join(data_parts))

def get_intervals_from_user(current_input_mode_ref, intervals_data):
    """
    Función para manejar la entrada y eliminación de intervalos por el usuario.
    intervals_data ahora se pasa por referencia y se modifica directamente.
    """
    
    print(f"\n--- Entrada/Gestión de Intervalos (Modo actual: {current_input_mode_ref['mode']}) ('k' para finalizar, 'V' para Valor Numérico, 'M' para Monzo, 'D' para eliminar) ---")
    i = len(intervals_data) + 1 # Comienza a contar desde el siguiente intervalo disponible

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
                    print("La lista de intervalos ya está vacía. No hay intervalos para eliminar.")
                    continue
                # Llama a la función de eliminación
                delete_interval(intervals_data, current_sort_order) # Pasa current_sort_order
                i = len(intervals_data) + 1 # Actualiza el contador después de la eliminación
                continue # Continúa el bucle de entrada/gestión
            elif new_mode_signal != current_input_mode_ref['mode']:
                continue
            
            if val is None and monzo is None: # Si la entrada fue para cambiar de modo o inválida
                continue

            original_input_representation = format_monzo_vector_angle_brackets(monzo, is_completely_factorized)

        else: # current_input_mode_ref['mode'] == 'V'
            input_prompt = f"Intervalo {i} (Valor Numérico): "
            input_received = input(input_prompt).strip()

            if input_received.lower() == 'k':
                break 
            elif input_received.upper() == 'V':
                if current_input_mode_ref['mode'] == 'V':
                    print("\nYa está en modo **Valor Numérico**.")
                else:
                    current_input_mode_ref['mode'] = 'V'
                    print("\nModo cambiado a **Valor Numérico**.")
                continue
            elif input_received.upper() == 'M':
                if current_input_mode_ref['mode'] == 'M':
                    print("\nYa está en modo **Monzo Fraccionario**.")
                else:
                    current_input_mode_ref['mode'] = 'M'
                    print("\nModo cambiado a **Monzo Fraccionario**.")
                continue
            elif input_received.upper() == 'D': # Nueva opción 'D' para eliminar
                if not intervals_data:
                    print("La lista de intervalos ya está vacía. No hay intervalos para eliminar.")
                    continue
                delete_interval(intervals_data, current_sort_order) # Pasa current_sort_order
                i = len(intervals_data) + 1 # Actualiza el contador después de la eliminación
                continue # Continúa el bucle de entrada/gestión

            val, monzo, is_completely_factorized = parse_value_input(input_received)
            original_input_representation = input_received
        
        if val is not None and monzo is not None and is_completely_factorized is not None:
            # Verifica si hay duplicados antes de agregar
            is_duplicate = False
            for _, existing_val, existing_monzo, _ in intervals_data:
                # Compara por valor y monzo para robustez
                if abs(existing_val - val) < 1e-9 and existing_monzo == monzo:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print(f"El intervalo '{original_input_representation}' (Valor: {val}) ya está en la lista. Por favor, ingrese otro intervalo.")
            else:
                intervals_data.append((original_input_representation, val, monzo, is_completely_factorized))
                i += 1
    # La función no devuelve nada, pero modifica intervals_data directamente.


def delete_interval(intervals_data, sort_order):
    """Permite al usuario seleccionar y eliminar un intervalo de la lista, ordenando la visualización."""
    if not intervals_data:
        print("\nNo hay intervalos presentes en la lista para eliminar.")
        return

    print("\n--- Eliminar Intervalo ---")
    
    # Crea una lista temporal de diccionarios para facilitar la clasificación y visualización
    display_list = []
    # Usa un conjunto para rastrear los monzos vistos en display_list para evitar duplicados en la vista de eliminación
    seen_monzos_in_display = set() 

    for idx, (original_str_repr, val, monzo_raw, is_completely_factorized) in enumerate(intervals_data):
        monzo_tuple = tuple(monzo_raw)
        if monzo_tuple not in seen_monzos_in_display:
            display_list.append({
                'original_index': idx, # Almacena el índice original para la eliminación real de intervals_data
                'original_str_repr': original_str_repr,
                'val': val, # Este es el valor de entrada original (usado para clasificar N.C. y formatear)
                'monzo_raw': monzo_raw,
                'is_completely_factorized': is_completely_factorized,
            })
            seen_monzos_in_display.add(monzo_tuple)


    # Clasifica la display_list basándose en el sort_order actual
    if sort_order == 'M':
        # Clasifica por estado de factorización (N.C. al final), luego por monzo
        display_list.sort(key=lambda x: (not x['is_completely_factorized'], [float(f) for f in x['monzo_raw']]))
        print("Intervalos actuales (clasificados por Monzo lexicográfico):")
    elif sort_order == 'C':
        # Clasifica por el 'val' (valor original) que siempre es preciso
        display_list.sort(key=lambda x: calculate_cents(x['val'])) # Clasifica por cents del valor original
        print("Intervalos actuales (clasificados por valor ascendente):")
    else: # Por defecto a C si sort_order es desconocido
        display_list.sort(key=lambda x: calculate_cents(x['val'])) # Usa cents del valor original para la clasificación predeterminada
        print(f"Advertencia: La opción de clasificación '{sort_order}' no reconocida. Clasificando por valor ascendente.")
    
    # Calcula los anchos máximos para una visualización consistente
    max_idx_len = len(str(len(display_list)))
    max_orig_str_len = len("Intervalo original")
    max_monzo_len = len("Monzo relativo")
    max_fact_len = len("Factorización")
    
    max_val_before_sep = 0
    max_val_after_sep = 0
    max_frac_str_len_val_col = 0 # Longitud máxima para la parte ;N/D en la columna Valor

    for item in display_list:
        max_orig_str_len = max(max_orig_str_len, len(item['original_str_repr']))
        max_monzo_len = max(max_monzo_len, len(format_monzo_vector_angle_brackets(item['monzo_raw'], item['is_completely_factorized'])))
        max_fact_len = max(max_fact_len, len(format_monzo_factorization(item['monzo_raw'], PRIMES, item['is_completely_factorized'])))
        
        # Calcula para la alineación del valor en este contexto de visualización
        val_for_display_calc = item['val'] # Siempre usa el valor original del campo 'val' para la visualización N.C.

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
    
    # Ancho total de la columna de Valor para la lista de visualización
    val_col_total_width = max(len("Valor"), max_val_before_sep + (1 if max_val_after_sep > 0 else 0) + max_val_after_sep + max_frac_str_len_val_col)


    # Imprime el encabezado para la lista de eliminación
    header_parts = [
        f"{'#':<{max_idx_len}}",
        f"{'Intervalo original':<{max_orig_str_len}}",
        f"{'Monzo relativo':<{max_monzo_len}}",
        f"{'Factorización':<{max_fact_len}}",
        f"{'Valor':<{val_col_total_width}}"
    ]
    header_line = " | ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))

    for i, item in enumerate(display_list):
        # Es crucial: Usa item['val'] (valor original) para formatear intervalos N.C.
        val_to_format_actual = item['val'] # Siempre usa el valor original del campo 'val' para la visualización N.C.

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
            f"{formatted_value:<{val_col_total_width}}" # Asegura que se use el ancho total de la columna aquí
        ]
        print(" | ".join(row_parts))

    while True:
        try:
            choice_str = input("Ingrese el número del intervalo a eliminar (o '0' para cancelar): ").strip()
            choice = int(choice_str)

            if choice == 0:
                print("Eliminación cancelada.")
                return
            
            if 1 <= choice <= len(display_list):
                # Obtiene el índice original de la lista clasificada para eliminar del intervals_data real
                original_index_to_delete = display_list[choice - 1]['original_index']
                removed_interval = intervals_data.pop(original_index_to_delete)
                print(f"Intervalo '{removed_interval[0]}' eliminado con éxito.")
                break
            else:
                print("Número inválido. Por favor, ingrese un número correspondiente a un intervalo o '0' para cancelar.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número.")

def is_monzo_rational(monzo_list):
    """Verifica si todos los exponentes en un Monzo son enteros."""
    return all(exp.denominator == 1 for exp in monzo_list)


def print_summary_table(intervals_data, sort_order='None'):
    """Imprime el resumen de los intervalos originales en formato tabular."""
    
    unique_intervals_summary = []
    # Usa un conjunto para rastrear los monzos únicos (como tuplas para hashabilidad)
    seen_monzos = set() 

    # Recopila datos y rastrea las longitudes máximas para la alineación
    temp_max_val_before_sep = 0
    temp_max_val_after_sep = 0
    max_frac_str_len = 0 # Longitud máxima de la parte ";numerador/denominador"

    for original_str_repr, val, monzo_raw, is_completely_factorized in intervals_data:
        monzo_tuple = tuple(monzo_raw)
        if monzo_tuple not in seen_monzos:
            calculated_value = calculate_monzo_value(monzo_raw, PRIMES)
            
            # --- Calcula las longitudes máximas para la alineación de esta tabla ---
            # Determina el valor a usar para el cálculo de la longitud de visualización
            val_for_display_calc = val if not is_completely_factorized else calculated_value

            # Para la parte decimal (antes de ';')
            decimal_part_str_temp = f"{val_for_display_calc:.5f}"
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(int_p_dec))
                temp_max_val_after_sep = max(temp_max_val_after_sep, len(frac_p_dec))
            else:
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(decimal_part_str_temp))
                temp_max_val_after_sep = max(temp_max_val_after_sep, 0)
            
            # Para la parte fraccionaria (después de ';')
            if is_monzo_rational(monzo_raw) and is_completely_factorized:
                try:
                    frac = Fraction(calculated_value).limit_denominator(10**6)
                    if frac.denominator != 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}/{frac.denominator}"))
                    elif frac.denominator == 1: # Caso entero, aún formatear como ';int'
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}"))
                except (OverflowError, ZeroDivisionError, ValueError):
                    pass # No tendrá parte fraccionaria
            # --- Fin del cálculo de longitud máxima ---

            unique_intervals_summary.append({
                'original_str_repr': original_str_repr,
                'monzo_raw': monzo_raw,
                'calculated_value': calculated_value,
                'val_original': val, # Almacena el valor original para la visualización N.C.
                'is_completely_factorized': is_completely_factorized
            })
            seen_monzos.add(monzo_tuple)
    
    if sort_order == 'M':
        # Clasifica por monzo, poniendo "N.C." al final
        unique_intervals_summary.sort(key=lambda x: (not x['is_completely_factorized'], [float(f) for f in x['monzo_raw']]))
    elif sort_order == 'C':
        # Clasifica por el valor original para intervalos N.C., valor calculado para los factorizables
        unique_intervals_summary.sort(key=lambda x: calculate_cents(x['val_original']) if not x['is_completely_factorized'] else calculate_cents(x['calculated_value']))

    headers = ["Original Interval", "Relative Monzo", "Factorization", "Value"]
    
    # Calcula los anchos de las columnas basándose en los datos + encabezados
    col_widths = {header: len(header) for header in headers}
    
    table_rows_data_for_formatting = [] # Almacena datos sin procesar, luego formatea para imprimir
    for interval_info in unique_intervals_summary:
        original_str_repr = interval_info['original_str_repr']
        monzo_raw = interval_info['monzo_raw']
        calculated_value = interval_info['calculated_value']
        is_completely_factorized = interval_info['is_completely_factorized']
        val_original_for_nc_display = interval_info['val_original'] # Para la visualización N.C.

        formatted_monzo_vector = format_monzo_vector_angle_brackets(monzo_raw, is_completely_factorized)
        formatted_factorization = format_monzo_factorization(monzo_raw, PRIMES, is_completely_factorized)
        
        # Determina el valor a pasar a format_value_for_alignment
        val_to_format_actual = val_original_for_nc_display if not is_completely_factorized else calculated_value

        # No formateamos el valor aquí, se hace en el bucle de impresión con los valores de alineación finales
        table_rows_data_for_formatting.append({
            'original_str_repr': original_str_repr,
            'formatted_monzo_vector': formatted_monzo_vector,
            'formatted_factorization': formatted_factorization,
            'val_to_format_actual': val_to_format_actual, # Esto contiene el valor float
            'monzo_raw': monzo_raw, # Mantiene el monzo sin procesar para la verificación de racionalidad en format_value_for_alignment
            'is_completely_factorized': is_completely_factorized # Mantiene el estado de factorización
        })

        # Actualiza el ancho mínimo para otras columnas basándose en los datos formateados
        col_widths[headers[0]] = max(col_widths[headers[0]], len(original_str_repr))
        col_widths[headers[1]] = max(col_widths[headers[1]], len(formatted_monzo_vector))
        col_widths[headers[2]] = max(col_widths[headers[2]], len(formatted_factorization))
        # El ancho de la columna de Valor necesita la longitud máxima total
        
    # Finaliza el ancho de la columna de Valor: temp_max_val_before_sep + (1 si temp_max_val_after_sep > 0 sino 0) + temp_max_val_after_sep + max_frac_str_len
    # Más el posible relleno
    col_widths[headers[3]] = max(len(headers[3]), temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len)


    # Imprime el encabezado
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    
    table_title = "TABLE OF ORIGINAL INTERVALS AND THEIR FACTORIZATION"
    # Asegura que los bordes de la tabla sean al menos tan largos como el título
    border_length = max(len(header_line), len(table_title))

    print("\n" + "=" * border_length)
    print(table_title.center(border_length))
    print("=" * border_length)
    print(header_line)
    print("-" * len(header_line))

    # Imprime las filas
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
                                f"{val_col_str_aligned:<{col_widths[headers[3]]}}"]) # Asegura que se use el ancho total de la columna aquí
        print(row_line)


def print_normalized_summary_table(normalization_base, intervals_data, sort_order='None'):
    """Imprime el resumen de los intervalos normalizados en formato tabular."""
    if normalization_base == 'w':
        print_summary_table(intervals_data, sort_order) # Vuelve a la tabla resumen original
        return

    unique_intervals_summary = []
    seen_monzos_original = set() 
    
    # Recopila datos y rastrea las longitudes máximas para la alineación
    temp_max_val_before_sep = 0
    temp_max_val_after_sep = 0
    max_frac_str_len = 0 # Longitud máxima de la parte ";numerador/denominador"


    for original_str_repr, val_original, monzo_raw_original, is_original_monzo_valid in intervals_data:
        monzo_tuple_original = tuple(monzo_raw_original)
        if monzo_tuple_original not in seen_monzos_original:
            # Pasa val_original a normalize_interval
            normalized_monzo, normalized_val_actual, is_normalized_monzo_valid_for_display = normalize_interval(val_original, monzo_raw_original, normalization_base, PRIMES, is_original_monzo_valid)
            
            # Para fines de visualización, siempre usa el valor numéricamente normalizado y sus cents
            display_value_norm = normalized_val_actual
            display_cents_norm = calculate_cents(normalized_val_actual)

            # --- Calcula las longitudes máximas para la alineación de esta tabla ---
            is_rational_norm_monzo = is_monzo_rational(normalized_monzo)
            
            # Para la parte decimal (antes de ';')
            decimal_part_str_temp = f"{display_value_norm:.5f}" if isinstance(display_value_norm, (float, int, Fraction)) else str(display_value_norm)
            if '.' in decimal_part_str_temp:
                int_p_dec, frac_p_dec = decimal_part_str_temp.split('.')
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(int_p_dec))
                temp_max_val_after_sep = max(temp_max_val_after_sep, len(frac_p_dec))
            else:
                temp_max_val_before_sep = max(temp_max_val_before_sep, len(decimal_part_str_temp))
                temp_max_val_after_sep = max(temp_max_val_after_sep, 0)
            
            # Para la parte fraccionaria (después de ';')
            if is_normalized_monzo_valid_for_display and is_rational_norm_monzo:
                try:
                    frac = Fraction(display_value_norm).limit_denominator(10**6)
                    if frac.denominator != 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}/{frac.denominator}"))
                    elif frac.denominator == 1:
                        max_frac_str_len = max(max_frac_str_len, len(f";{frac.numerator}"))
                except (OverflowError, ZeroDivisionError, ValueError):
                    pass
            # --- Fin del cálculo de longitud máxima ---

            unique_intervals_summary.append({
                'original_str_repr': original_str_repr,
                'normalized_monzo': normalized_monzo,
                'normalized_val': normalized_val_actual, # Valor de normalized_interval
                'display_value': display_value_norm, # Valor para visualización
                'normalized_cents': display_cents_norm, # Cents de normalized_interval
                'display_cents': f"{display_cents_norm:+.3f}", # Cents para visualización
                'is_normalized_monzo_valid': is_normalized_monzo_valid_for_display, # Validez del monzo normalizado para el formato
                'is_original_monzo_valid': is_original_monzo_valid # Validez del monzo original (para clasificación)
            })
            seen_monzos_original.add(monzo_tuple_original) # Agrega monzo original para verificación de unicidad

    if sort_order == 'M':
        # Clasifica por monzo, poniendo "N.C." al final. Clasificamos basándonos en la validez del monzo original
        unique_intervals_summary.sort(key=lambda x: (not x['is_original_monzo_valid'], [float(f) for f in x['normalized_monzo']]))
    elif sort_order == 'C':
        unique_intervals_summary.sort(key=lambda x: x['normalized_cents']) # Clasifica por normalized_cents que siempre usa el valor normalizado
    else:
        print(f"Advertencia: La opción de clasificación '{sort_order}' no reconocida. No se aplicó ninguna clasificación.")

    headers = ["Original Interval", "Relative Monzo Norm.", "Factorization Norm.", "Value Norm.", "Cents Norm."]

    # Calcula los anchos de las columnas basándose en los datos + encabezados
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
        
        # Usa la función de formato dedicada para la alineación
        value_display_str_aligned = format_value_for_alignment(
            display_value, 
            is_monzo_rational(normalized_monzo), # Verifica la racionalidad del monzo normalizado
            is_normalized_monzo_valid, # Verifica el estado de factorización del monzo normalizado
            temp_max_val_before_sep, temp_max_val_after_sep
        )
        
        cents_display_str_padded = f"{display_cents:+.3f}" if isinstance(display_cents, (float, int)) else str(display_cents) # Verificación defensiva

        table_rows_data_for_formatting.append([original_str_repr, normalized_monzo_vector, formatted_factorization, value_display_str_aligned, cents_display_str_padded])

        # Actualiza el ancho mínimo para otras columnas basándose en los datos formateados
        col_widths[headers[0]] = max(col_widths[headers[0]], len(original_str_repr))
        col_widths[headers[1]] = max(col_widths[headers[1]], len(normalized_monzo_vector))
        col_widths[headers[2]] = max(col_widths[headers[2]], len(formatted_factorization))
        # El ancho de la columna Valor Norm. se establece por la lógica de temp_max_val_before_sep/after_sep
        col_widths[headers[4]] = max(col_widths[headers[4]], len(cents_display_str_padded))

    # Finaliza el ancho de la columna Valor Norm.: temp_max_val_before_sep + (1 si temp_max_val_after_sep > 0 sino 0) + temp_max_val_after_sep + max_frac_str_len
    col_widths[headers[3]] = max(len(headers[3]), temp_max_val_before_sep + (1 if temp_max_val_after_sep > 0 else 0) + temp_max_val_after_sep + max_frac_str_len)

    # Imprime el encabezado
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    
    table_title = f"TABLE OF LOGARITHMIC NORMALIZED INTERVALS to base {normalization_base}"
    # Asegura que los bordes de la tabla sean al menos tan largos como el título
    border_length = max(len(header_line), len(table_title))

    print("\n" + "=" * border_length)
    print(table_title.center(border_length))
    print("=" * border_length)
    print(header_line)
    print("-" * len(header_line))

    # Imprime las filas
    for row_info in table_rows_data_for_formatting:
        row_line = " | ".join([f"{row_info[0]:<{col_widths[headers[0]]}}",
                                f"{row_info[1]:<{col_widths[headers[1]]}}",
                                f"{row_info[2]:<{col_widths[headers[2]]}}",
                                f"{row_info[3]:<{col_widths[headers[3]]}}", # Ya alineado por format_value_for_alignment
                                f"{row_info[4]:<{col_widths[headers[4]]}}"])
        print(row_line)

# --- PROGRAMA PRINCIPAL ---

# Función para obtener entrada 'r' válida, aislada para robustez
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
    # Estas variables almacenarán los rangos para a y b
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
    
    # Entrada de intervalos
    get_intervals_from_user(current_input_mode_ref, intervals_data)

    if not intervals_data:
        print("No se ingresaron intervalos. Saliendo.")
        return

    # Solicita los parámetros de temperamento después de la entrada de intervalos
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

    # Imprime tablas
    print_tables(current_min_a, current_max_a, current_min_b, current_max_b, intervals_data, current_sort_order)
    
    # Imprime tabla resumen
    print_summary_table(intervals_data, current_sort_order)


    # --- Bucle para cambiar la clasificación o ingresar/eliminar nuevos intervalos ---
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
            else: # Es un entero válido > 1
                print_normalized_summary_table(chosen_r_val, intervals_data, current_sort_order) # Usa la nueva función de tabla
            
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