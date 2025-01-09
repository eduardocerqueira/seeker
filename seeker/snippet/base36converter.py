#date: 2025-01-09T17:02:07Z
#url: https://api.github.com/gists/e983334bbb762de03ddb42ef07a77fd4
#owner: https://api.github.com/users/AdamBojek

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def number_conversion(n : str, input_base : int, output_base : int) -> str :
    n = n.lstrip('0')
    n = n.upper()

    try:
        #prostszy zapis not all(x in characters[:input_base] for x in n):
        if not all(map(lambda x: True if x in characters[0:input_base] else False, n)):
            raise Exception("Nieprawidłowe znaki w liczbie wejściowej")
        if input_base <= 1 or input_base > 36 or output_base <= 1 or output_base > 36:
            raise Exception("Nieprawidłowa podstawa")
    except BaseException as error:
        return error

    dec = 0
    power = 1
    for i in range(len(n)-1, -1, -1):
        dec += characters.index(n[i]) * power
        power *= input_base

    output = ""
    while dec > 0:
        output += characters[(dec % output_base)]
        dec //= output_base

    return output[::-1]