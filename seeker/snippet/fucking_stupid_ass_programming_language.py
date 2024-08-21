#date: 2024-08-21T17:06:09Z
#url: https://api.github.com/gists/14e60b0045943a7ed5b4af0c81013dde
#owner: https://api.github.com/users/blockchain200

class GenAlphaInterpreter:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.namespaces = {}
    
    def execute(self, code):
        lines = code.split('\n')
        for line in lines:
            self.execute_line(line.strip())
    
    def execute_line(self, line):
        if line.startswith("skibidi"):
            self.handle_variable(line)
        elif line.startswith("gyat"):
            self.handle_function(line)
        elif line.startswith("sigma"):
            self.handle_class(line)
        elif line.startswith("rizz"):
            self.handle_print(line)
        elif line.startswith("ohio"):
            self.handle_namespace(line)
    
    def handle_variable(self, line):
        parts = line.split('=')
        var_name = parts[0].replace("skibidi", "").strip()
        var_value = eval(parts[1].strip())
        self.variables[var_name] = var_value
    
    def handle_function(self, line):
        parts = line.split(':')
        func_name = parts[0].replace("gyat", "").strip()
        func_body = parts[1].strip()
        self.functions[func_name] = func_body
    
    def handle_class(self, line):
        parts = line.split(':')
        class_name = parts[0].replace("sigma", "").strip()
        class_body = parts[1].strip()
        self.variables[class_name] = type(class_name, (), {})
    
    def handle_print(self, line):
        message = line.replace("rizz", "").strip()
        print(eval(message, {}, self.variables))
    
    def handle_namespace(self, line):
        parts = line.split(':')
        namespace_name = parts[0].replace("ohio", "").strip()
        namespace_body = parts[1].strip()
        self.namespaces[namespace_name] = namespace_body

# Example usage:
code = """
skibidi x = 5
rizz x
gyat hello:
    rizz "Hello, world!"
sigma MyClass:
    skibidi y = 10
ohio my_space:
    rizz "In Ohio namespace"
"""

interpreter = GenAlphaInterpreter()
interpreter.execute(code)
