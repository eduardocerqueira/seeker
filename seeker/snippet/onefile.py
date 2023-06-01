#date: 2023-06-01T16:52:33Z
#url: https://api.github.com/gists/a9d81c4f07c5b2cbc21fff5aa0695a15
#owner: https://api.github.com/users/emresvd

import sys
import os
import autopep8

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    code = f.read()

def remove_comma_in_imports(code):
    new_code = []
    for line in code.splitlines():
        if line.startswith('import'):
            new_code.append(line.replace(',', '\nimport '))
        else:
            new_code.append(line)
    return autopep8.fix_code('\n'.join(new_code))


def sort_imports(code):
    import_lines = []
    for line in code.splitlines():
        if line.startswith('import'):
            import_lines.append(line)
    sorted_import_lines = []
    sorted_import_lines += import_lines
    sorted_import_lines.sort(key=len)
    sorted_import_lines.reverse()
    return code.replace('\n'.join(import_lines), '\n'.join(sorted_import_lines))

code = remove_comma_in_imports(code)
code = sort_imports(code)

def get_module_path(module_name):
    module_path = os.path.join(os.path.dirname(file_path), module_name.replace(".", os.sep))
    module_path = os.path.abspath(module_path)
    module_path = os.path.normpath(module_path)
    module_path += '.py'
    return module_path

def putcode(module_path, line, module_name, import_=False):
    global code, added_module_names
    if module_name in added_module_names:
        code = code.replace(line, "")
        return
    with open(module_path, "r") as f:
        module_code = f.read()
    code = code.replace(line, module_code)
    if import_:
        for i in ["\n", " ", "=", "("]:
            code = code.replace(f"{i}{module_name}.", i)
    added_module_names.append(module_name)


def add_code_from_line(line, import_=True):
    if import_:
        module_name = line.replace("import", "").strip()
    else:
        module_name = line.replace("from", "").split("import")[0].strip()

    if module_name.startswith('.'):
        module_name = module_name[1:]

    module_path = get_module_path(module_name)

    if os.path.isdir(module_path.replace(".py", "")):
        module_path = os.path.join(module_path.replace(".py", ""), "__init__.py")

    if os.path.isfile(module_path):
        project_modules.append(module_path)
        putcode(module_path, line, module_name, import_=import_)

added_module_names = []

while True:
    project_modules = []
    for line in code.splitlines():
        if line.startswith('import') or line.startswith('from'):
            add_code_from_line(line, import_=line.startswith('import'))
    if not project_modules:
        break

code = autopep8.fix_code(code)
print(code)
