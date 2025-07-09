#date: 2025-07-09T16:57:02Z
#url: https://api.github.com/gists/87d75448b52d82ca06f681450ab05b3e
#owner: https://api.github.com/users/basperheim

import os
import re
import argparse
from collections import defaultdict

# Step 0: Utility Functions
def find_parent_method(lines, call_line_num):
    """
    Walk upward from the given line number to find the method that encloses a method call.
    Returns a tuple of (line_num, method_signature) or None.
    """
    method_pattern = re.compile(
        r'(public|private|protected)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(:\s*[a-zA-Z0-9_<>\[\]| ]+)?\s*\{'
    )
    for i in range(call_line_num - 2, -1, -1):
        line = lines[i].strip()
        match = method_pattern.match(line)
        if match:
            method_name = match.group(2)
            method_args = match.group(3).strip()
            method_sig = f"{method_name}({method_args})"
            return (i + 1, method_sig)
    return None

def find_enclosing_condition(lines, call_line_num, parent_method_line):
    """
    Walk upward from the call line up to the method start to find enclosing condition/loop/switch.
    Returns the first matching block pattern as (line_num, line) or None.
    """
    block_pattern = re.compile(
        r'^(if|else if|else|for|while|switch)\b[^\{]*\{?'
    )
    brace_depth = 0
    for i in range(call_line_num - 2, parent_method_line - 2, -1):
        line = lines[i].strip()
        brace_depth += line.count('}')
        brace_depth -= line.count('{')
        if brace_depth < 0:
            brace_depth = 0
        if brace_depth == 0 and block_pattern.match(line):
            return (i + 1, line)
    return None

def find_enclosing_blocks(lines, call_line_num):
    """
    Scan upward from a line to find enclosing block condition (if/for/etc) and method signature.
    Returns: (enclosing_condition, enclosing_method) as tuples with line numbers.
    """
    brace_depth = 0
    enclosing_condition = None
    enclosing_method = None
    for i in range(call_line_num - 2, -1, -1):
        line = lines[i].strip()
        brace_depth += line.count('}')
        brace_depth -= line.count('{')
        if brace_depth < 0:
            brace_depth = 0
        if brace_depth == 0:
            if re.match(r'(if|else if|else|for|while|switch)\b[^\{]*\{?$', line) and not enclosing_condition:
                enclosing_condition = (i + 1, line)
            m2 = re.match(r'(public|private|protected)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', line)
            if m2 and not enclosing_method:
                method_signature = f"{m2.group(2)}({m2.group(3).strip()})"
                enclosing_method = (i + 1, method_signature)
        if enclosing_condition and enclosing_method:
            break
    return enclosing_condition, enclosing_method

# Step 1: Component File Discovery
def find_component_files(directory, match_string):
    """
    Recursively walk through directory and return .ts files that include @Component
    and a selector that includes the match_string.
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ts'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '@Component' in content:
                            selector_match = re.search(r"selector:\s*['\"]([^'\"]+)['\"]", content)
                            if selector_match and match_string in selector_match.group(1):
                                print(file_path)
                                matching_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return matching_files

# Step 2: Extract JSDoc for a given method line
def extract_jsdoc(lines, method_line_num):
    """
    Given the line number of a method declaration, find preceding JSDoc block (/** ... */)
    and return it as a cleaned-up string. Returns None if not found.
    """
    i = method_line_num - 2
    jsdoc_lines = []
    while i >= 0:
        line = lines[i].rstrip()
        if line.strip() == '':
            i -= 1
            continue
        if line.strip().endswith('*/'):
            while i >= 0:
                line = lines[i].rstrip()
                jsdoc_lines.append(line)
                if line.strip().startswith('/**'):
                    break
                i -= 1
            jsdoc_clean = []
            for l in reversed(jsdoc_lines):
                l = l.strip()
                l = re.sub(r'^/\*\*|\*/$', '', l)
                l = re.sub(r'^\*\s?', '', l)
                if l:
                    jsdoc_clean.append("    " + l)
            return '\n'.join(jsdoc_clean)
        elif not line.strip().startswith('*') and not line.strip().startswith('/**'):
            break
        i -= 1
    return None

# Step 3: Extract Method Declarations (Blocks)
def extract_method_blocks(lines):
    """
    Return a list of method blocks found in the file.
    Each block includes name, args, start line, signature, and optional end line.
    """
    method_blocks = []
    method_pattern = re.compile(r'^(?!\s*ng)[^\n]*\([^\n]*\)[^\n]*:[^\n]*\{')
    inside = False
    brace_stack = []
    current = None
    for i, line in enumerate(lines):
        if not inside:
            m = method_pattern.match(line)
            if m:
                raw_declaration = m.group(0).strip()
                if (
                    '.then' in raw_declaration or
                    'this.' in raw_declaration or
                    'if (' in raw_declaration or
                    'error:' in raw_declaration or
                    '=> {' in raw_declaration
                ):
                    continue
                access_match = re.match(r'^(public|private|protected)?\s*(async\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', raw_declaration)
                if access_match:
                    access_modifier = access_match.group(1) or 'public'
                    method_name = access_match.group(3)
                    method_args = access_match.group(4).strip()
                else:
                    print(f"⚠️  Couldn't parse method declaration: {raw_declaration}")
                    continue
                sig = f"{access_modifier} {method_name}({method_args})"
                print(f"\n\033[94m[FOUND]\033[0m {sig}  \033[90m(Line {i+1})\033[0m")
                if '{' in line:
                    inside = True
                    current = {
                        'name': method_name,
                        'args': method_args,
                        'start': i + 1,
                        'signature': sig
                    }
                    brace_stack = [1]
        else:
            opens = line.count('{')
            closes = line.count('}')
            brace_stack[-1] += opens
            brace_stack[-1] -= closes
            if brace_stack[-1] <= 0:
                inside = False
                current['end'] = i + 1
                method_blocks.append(current)
                current = None
                brace_stack = []
    return method_blocks

# Step 4: Extract method calls + attach JSDoc, context
def extract_methods_and_calls(file_path):
    """
    Extract all declared methods and actual call sites.
    Also capture jsdoc, enclosing if/switch, and method nesting.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    method_blocks = extract_method_blocks(lines)
    methods = {}
    for mb in method_blocks:
        jsdoc = extract_jsdoc(lines, mb['start'])
        methods[mb['name']] = {
            'args': mb['args'],
            'decl_line': mb['start'],
            'calls': [],
            'jsdoc': jsdoc
        }
    for i, line in enumerate(lines):
        for method_name in methods:
            if re.search(rf'\b{re.escape(method_name)}\s*\(', line) and (i + 1) != methods[method_name]['decl_line']:
                parent = find_parent_method_block(method_blocks, i + 1)
                cond = find_enclosing_condition(lines, i + 1, parent[0]) if parent else None
                methods[method_name]['calls'].append({
                    'line_num': i + 1,
                    'line': line.strip(),
                    'condition': cond,
                    'parent_method': parent
                })
    return methods

# Step 5: Print Results
def print_results(file_path, methods):
    """
    Nicely print parsed method metadata, JSDoc, and usage calls with context.
    """
    print(f"\n--- Analyzing {file_path} ---\n")
    if not methods:
        print("No methods found.")
        return
    for method_name, data in methods.items():
        decl_line = data['decl_line']
        args = data['args']
        print(f"\033[94mMethod:\033[0m {method_name}({args}) \033[90m[L{decl_line}]\033[0m")
        if data.get('jsdoc'):
            print(f"  \033[93mJS Doc:\033[0m\n{data['jsdoc']}")
        if data['calls']:
            print(f"  \033[92mCalls:\033[0m")
            for call in data['calls']:
                print(f"    \033[90mL{call['line_num']}:\033[0m {call['line']}")
                if call['condition']:
                    print(f"      └── Inside: {call['condition'][1]} \033[90m[L{call['condition'][0]}]\033[0m")
                if call['parent_method']:
                    print(f"      └── Parent Method: {call['parent_method'][1]} \033[90m[L{call['parent_method'][0]}]\033[0m")
        else:
            print(f"  \033[91mNo calls found.\033[0m")
        print()

# Step 6: Determine method enclosing block
def find_parent_method_block(method_blocks, call_line):
    """
    From all known method blocks, find the method that contains the given call line.
    Returns (start_line, signature) or None.
    """
    candidates = [mb for mb in method_blocks if mb['start'] <= call_line <= mb.get('end', 10**6)]
    if candidates:
        chosen = max(candidates, key=lambda mb: mb['start'])
        return (chosen['start'], chosen['signature'])
    return None

# Step 7: CLI entrypoint
def main():
    parser = argparse.ArgumentParser(description='Search Angular components and extract methods and their call sites.')
    parser.add_argument('--match', required=True, help='Partial component selector to match')
    args = parser.parse_args()
    directory = 'frontend/src/app'

    matching_files = find_component_files(directory, args.match) # Step #1

    if not matching_files:
        print("No matching files found.")
        return
    for file in matching_files:
        methods = extract_methods_and_calls(file) # Step #4
        print_results(file, methods)

if __name__ == '__main__':
    main()