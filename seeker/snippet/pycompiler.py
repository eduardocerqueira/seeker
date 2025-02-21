#date: 2025-02-21T17:10:43Z
#url: https://api.github.com/gists/3f7eb469e68c4e3824ac09b1dbc95525
#owner: https://api.github.com/users/SWORDIntel

#!/usr/bin/env python3
#pip install uncompyle6
import os
import subprocess
import py_compile
import compileall
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

class CompilerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python Compiler/Decompiler")
        self.create_widgets()

    def create_widgets(self):
        # File selection section
        tk.Label(self, text="Select Python File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.file_var = tk.StringVar()
        self.file_entry = tk.Entry(self, textvariable=self.file_var, width=50)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        # Buttons for operations
        tk.Button(self, text="Compile Selected File", command=self.compile_selected).grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(self, text="Compile All Files in Directory", command=self.compile_all).grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(self, text="Decompile Selected .pyc", command=self.decompile_selected).grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        # Log and console output area
        tk.Label(self, text="Log Output:").grid(row=4, column=0, sticky="nw", padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(self, width=80, height=20)
        self.log_text.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        print(message)

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Python File", filetypes=[("Python Files", "*.py"), ("Compiled Python Files", "*.pyc")])
        if file_path:
            self.file_var.set(file_path)

    def compile_selected(self):
        file_path = self.file_var.get().strip()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error", "Please select a valid Python file.")
            return

        # If the file is a .py, compile it
        if file_path.endswith(".py"):
            try:
                # Compile to a .pyc in the same directory, with same base name.
                cfile = os.path.join(os.path.dirname(file_path), os.path.basename(file_path) + "c")
                py_compile.compile(file_path, cfile=cfile, doraise=True)
                self.log(f"Compiled {file_path} to {cfile}")
            except py_compile.PyCompileError as e:
                self.log(f"Compilation error: {e.msg}")
        else:
            self.log("Selected file is not a .py file.")

    def compile_all(self):
        # Ask user for a directory
        directory = filedialog.askdirectory(title="Select Directory to Compile")
        if not directory:
            return

        self.log(f"Compiling all Python files in directory: {directory}")
        # compileall.compile_dir returns True if successful; it writes output to sys.stdout.
        success = compileall.compile_dir(directory, force=True, quiet=1)
        if success:
            self.log("Compilation of all files completed successfully.")
        else:
            self.log("Some files failed to compile.")

    def decompile_selected(self):
        # The file should be a .pyc
        file_path = self.file_var.get().strip()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error", "Please select a valid .pyc file for decompilation.")
            return

        if not file_path.endswith(".pyc"):
            messagebox.showerror("Error", "Please select a .pyc file to decompile.")
            return

        # Attempt to decompile using uncompyle6
        self.log(f"Decompiling {file_path} ...")
        try:
            # Run uncompyle6 and capture output
            result = subprocess.run(["uncompyle6", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                self.log("Decompiled code:")
                self.log(result.stdout)
            else:
                self.log("Error during decompilation:")
                self.log(result.stderr)
        except FileNotFoundError:
            self.log("uncompyle6 is not installed or not found in PATH. Please install it using 'pip install uncompyle6'.")

if __name__ == "__main__":
    app = CompilerGUI()
    app.mainloop()
