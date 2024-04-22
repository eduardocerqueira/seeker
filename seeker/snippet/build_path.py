#date: 2024-04-22T17:02:38Z
#url: https://api.github.com/gists/04eebb9dd557c7f5bf101f1f688e2aa4
#owner: https://api.github.com/users/todaatsushi

import ast
import sys

def get_test_info(test_name: str, file_path: str) -> tuple[str, str | None, str | None]:
    with open(file_path) as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            containing_class = None
            for ancestor in ast.walk(tree):
                if isinstance(ancestor, ast.ClassDef) and node.lineno > ancestor.lineno:
                    containing_class = ancestor.name
                    break
            return file_path, containing_class, test_name

        elif isinstance(node, ast.ClassDef) and node.name == test_name:
            return file_path, test_name, None

    raise NotImplementedError("unreachable")

def clean_obj_name(obj: str) -> str:
    clean = obj.replace("def ", "").replace("class ", "")
    if "(" in clean:
        test, _ = clean.split("(")
    else:
        test = clean
    return test

if __name__ == "__main__":
    obj_name = clean_obj_name(sys.argv[1])
    file_path = sys.argv[2]

    path, klass, test = get_test_info(obj_name, file_path)

    res = path
    if klass:
        res += f"::{klass}"
    if test:
        res += f"::{test}"
    print(res)