#date: 2024-08-22T16:56:45Z
#url: https://api.github.com/gists/6bfa515ddc96397531af7756345af739
#owner: https://api.github.com/users/miller-itsec

import re
import math
import argparse
from collections import Counter

"""
This script is designed to detect obfuscated scripts in various scripting languages, including VBS, JavaScript, PowerShell, and CMD (Batch). 

### Algorithm Overview:
1. **Entropy Calculation**: Measures the randomness of the script's content. High entropy may indicate obfuscation.
2. **Variable Naming Score**: Analyzes the pattern of variable names. Nonsense or random variable names often signal obfuscation.
3. **Operator/Command Score**: Compares the frequency of common and uncommon operators or commands. Unusual usage patterns may suggest obfuscation.
4. **String/Math Operations Score**: Checks the frequency of string concatenations and math operations, which are often used in obfuscation techniques.
5. **Statement Length Score**: Examines the length of code statements. Very long statements are often found in obfuscated code.
6. **N-Gram Analysis Score**: Looks at the frequency of n-grams (sequences of n characters) to identify unusual patterns typical in obfuscated code.
7. **Character Frequency Analysis**: Analyzes the frequency of characters, especially focusing on non-alphanumeric characters, which are often overrepresented in obfuscated scripts.
8. **Weighted Scoring**: The algorithm combines these scores using weights that can be adjusted based on the scripting language profile to better reflect the obfuscation techniques commonly used in that language.

### CLI Usage:
The script accepts the scripting language and the file to be analyzed as command-line arguments. Based on the selected language, the script will apply the corresponding profile and determine whether the script appears obfuscated.
"""

# Define profiles for different scripting languages
profiles = {
    'vbs': {
        'common_operators': {'Dim', 'Set', 'If', 'Then', 'Else', 'End', 'Sub', 'Function', 'For', 'Next', 'Do', 'While', 'WScript', 'CreateObject', 'Call', 'On Error', 'Goto', 'Exit', 'MsgBox', 'InputBox'},
        'weights': {'entropy': 0.2, 'var_score': 0.2, 'op_cmd_score': 0.2, 'stmt_len_score': 0.1, 'str_math_score': 0.1, 'ngram_score': 0.1, 'char_freq_score': 0.1},
    },
    'js': {
        'common_operators': {'var', 'let', 'const', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'new', 'eval', 'document', 'window', 'this', 'catch', 'try', 'typeof', 'instanceof', 'import', 'export'},
        'weights': {'entropy': 0.25, 'var_score': 0.15, 'op_cmd_score': 0.2, 'stmt_len_score': 0.1, 'str_math_score': 0.15, 'ngram_score': 0.1, 'char_freq_score': 0.05},
    },
    'powershell': {
        'common_operators': {'Get-Command', 'Set-Variable', 'Invoke-Expression', 'New-Object', 'ForEach-Object', 'If', 'Else', 'While', 'Do', 'Function', 'Write-Output', 'Read-Host', 'Add-Type', 'Start-Process', 'Invoke-Command'},
        'weights': {'entropy': 0.3, 'var_score': 0.1, 'op_cmd_score': 0.25, 'stmt_len_score': 0.1, 'str_math_score': 0.1, 'ngram_score': 0.1, 'char_freq_score': 0.05},
    },
    'cmd': {
        'common_operators': {'if', 'else', 'for', 'goto', 'echo', 'set', 'pause', 'exit', 'rem', 'call', 'shift', 'cls', 'type', 'start', 'choice', 'setlocal', 'endlocal'},
        'weights': {'entropy': 0.2, 'var_score': 0.15, 'op_cmd_score': 0.2, 'stmt_len_score': 0.15, 'str_math_score': 0.1, 'ngram_score': 0.1, 'char_freq_score': 0.1},
    },
}

def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = float(x) / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

def variable_naming_score(code):
    variable_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    variables = variable_pattern.findall(code)
    nonsense_vars = [var for var in variables if re.match(r'^[a-z]{1,2}\d{2,}|[a-zA-Z0-9_]{8,}$', var)]
    return len(nonsense_vars) / len(variables) if variables else 0

def operator_command_score(code, common_operators):
    tokens = "**********"
    uncommon_tokens = "**********"
    return len(uncommon_tokens) / len(tokens) if tokens else 0

def string_math_operations_score(code):
    string_ops = len(re.findall(r'[+]', code))
    math_ops = len(re.findall(r'[-*/%]', code))
    return (string_ops + math_ops) / len(code) if len(code) else 0

def statement_length_score(code):
    statements = re.split(r'[;{}]', code)
    long_statements = [stmt for stmt in statements if len(stmt) > 80]
    return len(long_statements) / len(statements) if statements else 0

def ngram_analysis_score(code, n=3):
    ngrams = [code[i:i+n] for i in range(len(code)-n+1)]
    common_patterns = Counter(ngrams)
    unusual_ngrams = [ngram for ngram, count in common_patterns.items() if count == 1]
    return len(unusual_ngrams) / len(ngrams) if ngrams else 0

def char_frequency_analysis(code):
    total_chars = len(code)
    char_freq = Counter(code)
    freq_ratios = {char: freq / total_chars for char, freq in char_freq.items()}
    target_chars = [char for char, ratio in freq_ratios.items() if ratio > 0.1 and not char.isalnum()]
    return len(target_chars) / len(char_freq) if char_freq else 0

def is_obfuscated(code, profile):
    entropy = calculate_entropy(code)
    var_score = variable_naming_score(code)
    op_cmd_score = operator_command_score(code, profile['common_operators'])
    stmt_len_score = statement_length_score(code)
    str_math_score = string_math_operations_score(code)
    ngram_score = ngram_analysis_score(code)
    char_freq_score = char_frequency_analysis(code)
    
    # Apply weights based on the profile
    obfuscation_score = (
        entropy * profile['weights']['entropy'] +
        var_score * profile['weights']['var_score'] +
        op_cmd_score * profile['weights']['op_cmd_score'] +
        stmt_len_score * profile['weights']['stmt_len_score'] +
        str_math_score * profile['weights']['str_math_score'] +
        ngram_score * profile['weights']['ngram_score'] +
        char_freq_score * profile['weights']['char_freq_score']
    )
    
    return obfuscation_score > 0.6

def main():
    parser = argparse.ArgumentParser(description='Detect obfuscated scripts.')
    parser.add_argument('language', choices=['vbs', 'js', 'powershell', 'cmd'], help='The scripting language of the code.')
    parser.add_argument('file', help='The file containing the code to analyze.')
    
    args = parser.parse_args()
    
    with open(args.file, 'r') as file:
        code = file.read()
    
    profile = profiles[args.language]
    result = is_obfuscated(code, profile)
    
    if result:
        print(f"The {args.language.upper()} script appears to be obfuscated.")
    else:
        print(f"The {args.language.upper()} script does not appear to be obfuscated.")

if __name__ == "__main__":
    main()

ear to be obfuscated.")

if __name__ == "__main__":
    main()

