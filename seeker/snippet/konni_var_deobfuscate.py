#date: 2024-01-26T16:47:29Z
#url: https://api.github.com/gists/de46604ddc5803ca1e8681a1fc9217ab
#owner: https://api.github.com/users/SatyenderYadav

import re

def find_and_xor_patterns(code):
    patterns_by_variable = {}

    # Define a pattern to match the AND operation with Chr() in the code
   
    pattern = re.compile(r'(\w+)\s*=\s*\1\s*&\s*Chr\((\d+)\s*Xor\s*(\d+)\)')


    # Find all matches in the code
    matches = pattern.finditer(code)

    # Iterate over matches and organize patterns by variable
    for match in matches:
        variable = match.group(1)
        num1 = match.group(2)
        num2 = match.group(3)
        # Replace & with + and Xor with ^
        pattern_str = f"chr({num1} ^ {num2})"
        
        # Add the pattern to the corresponding variable list
        if variable in patterns_by_variable:
            patterns_by_variable[variable].append(pattern_str)
        else:
            patterns_by_variable[variable] = [pattern_str]

    # Join patterns with + for each variable
    for variable, patterns in patterns_by_variable.items():
        patterns_by_variable[variable] = ' + '.join(patterns)

    return patterns_by_variable

# Add Obfuscated Code Without any Changes
obfuscated_code = 
"""
"""
patterns_by_variable = find_and_xor_patterns(obfuscated_code)


for variable, pattern in patterns_by_variable.items():
    print(f"Variable: {variable}")
    print(eval(pattern))
    print()  
