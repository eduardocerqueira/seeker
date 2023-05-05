#date: 2023-05-05T16:38:20Z
#url: https://api.github.com/gists/317aa9d3a98ad0ac2be7f6ecfb430a9a
#owner: https://api.github.com/users/secemp9

import ast
import astunparse
from ast import NodeTransformer, fix_missing_locations
import tkinter as tk


class TkinterImportTransformer(NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == 'tkinter' and node.names[0].name == '*':
            return ast.Import(names=[ast.alias(name='tkinter', asname='tk')])
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id not in {'tkinter', 'tk'}:
            for attr_name in dir(tk):
                if node.func.id == attr_name:
                    node.func.id = f"tk.{attr_name}"
                    break
        return node


def correct_tkinter_import_usage(code):
    tree = ast.parse(code)
    transformed_tree = TkinterImportTransformer().visit(tree)
    fix_missing_locations(transformed_tree)
    corrected_code = astunparse.unparse(transformed_tree)
    return corrected_code


def retrieve_text():
    example = text_widget.get("1.0", "end-1c")
    corrected_code = correct_tkinter_import_usage(example)

    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, corrected_code)


root = tk.Tk()

text_widget = tk.Text(root)
text_widget.pack()

button = tk.Button(root, text="Fix Tk import", command=retrieve_text)
button.pack()


def copy_to_clipboard():
    root.clipboard_clear()
    root.clipboard_append(text_widget.get("1.0", tk.END))

button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
button.pack()
root.mainloop()

root.mainloop()