#date: 2025-11-10T16:48:05Z
#url: https://api.github.com/gists/d128acae09a28108137024820b6cb18b
#owner: https://api.github.com/users/Arsalan-Jafarnezhad

def national_code_valid(national_code: str) -> bool:
    if len(national_code) == 10:
        if national_code.isnumeric():
            length = len(national_code)
            counter = 0
            for index in range(len(national_code) - 1):
                counter += int(national_code[index]) * length
                length -= 1
            counter %= 11
            last_index = int(national_code[-1])
            if counter < 2:
                if counter == last_index:
                    return True
            elif counter - 11 == last_index:
                return True
    return False