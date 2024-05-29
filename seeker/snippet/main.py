#date: 2024-05-29T16:55:04Z
#url: https://api.github.com/gists/9cebf7f5a9d1c8a9409008015e223fb6
#owner: https://api.github.com/users/mypy-play

import re
import sys


class AST:
    def __init__(self, token_nuber: "**********": list[object]):
        self.token_nuber = "**********"
        self.args = args
        self.code = []


def tokenizer(program: "**********":
    return re.findall(r"[\(\)]|\"[^\"]*?\"|\'[^\']*?'|[\w\-+]+|!=|>=|\S", program)


# highlighting the token
def beautiful_token(tokens: "**********": int) -> str:
    start_index = max(number - 5, 0)
    end_index = "**********"
    ret = ""
    hret = "\r\n"
    for token in tokens[start_index: "**********":
        ret += "**********"
    hret += "**********"
    ret += "**********"
    for token in tokens[number + 1 : "**********":
        ret += "**********"
    return ret + hret


def build_ast(tokens: "**********":
    def build_ast_recursive(current_position):
        args = []
        pos = current_position
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"p "**********"o "**********"s "**********"i "**********"t "**********"i "**********"o "**********"n "**********"  "**********"< "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"a "**********"n "**********"d "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"[ "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"p "**********"o "**********"s "**********"i "**********"t "**********"i "**********"o "**********"n "**********"] "**********"  "**********"! "**********"= "**********"  "**********"" "**********") "**********"" "**********": "**********"
            token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"= "**********"= "**********"  "**********"" "**********"( "**********"" "**********": "**********"
                subtree, current_position = build_ast_recursive(current_position + 1)
                args.append(subtree)
            else:
                args.append(token)
            current_position += 1
        return AST(pos, args), current_position

    root, _ = build_ast_recursive(0)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"_ "**********"  "**********"! "**********"= "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"_ "**********"  "**********"> "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
            print("unexpected end of file")
        else:
            print("Unexpected token: "**********"
            print(beautiful_token(tokens, _))
        exit(1)
    return root


def translate(tokens: "**********": AST) -> str:
    global_data = []

    def t_assert(q: bool, text: str, ast: AST):
        if not q:
            print(text + ":")
            print(beautiful_token(tokens, ast.token_nuber))
            exit(1)

    rnd_lable_iter = 0

    def rnd_lable():
        nonlocal rnd_lable_iter
        rnd_lable_iter += 1
        return "lable_" + str(rnd_lable_iter)

    def t_is(token: "**********": str):
        assert type in ("variable", "number", "string"), "E95"
        return (
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"( "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"" "**********"v "**********"a "**********"r "**********"i "**********"a "**********"b "**********"l "**********"e "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"e "**********". "**********"m "**********"a "**********"t "**********"c "**********"h "**********"( "**********"r "**********"" "**********"^ "**********"[ "**********"A "**********"- "**********"Z "**********"a "**********"- "**********"z "**********"] "**********"[ "**********"A "**********"- "**********"Z "**********"a "**********"- "**********"z "**********"0 "**********"- "**********"9 "**********"] "**********"* "**********"$ "**********"" "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********") "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"o "**********"r "**********"  "**********"( "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"" "**********"n "**********"u "**********"m "**********"b "**********"e "**********"r "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"e "**********". "**********"m "**********"a "**********"t "**********"c "**********"h "**********"( "**********"r "**********"" "**********"^ "**********"( "**********"0 "**********"| "**********"- "**********"? "**********"[ "**********"1 "**********"- "**********"9 "**********"] "**********"[ "**********"0 "**********"- "**********"9 "**********"] "**********"* "**********") "**********"$ "**********"" "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********") "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"o "**********"r "**********"  "**********"( "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"" "**********"s "**********"t "**********"r "**********"i "**********"n "**********"g "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"e "**********". "**********"m "**********"a "**********"t "**********"c "**********"h "**********"( "**********"r "**********"' "**********"^ "**********"( "**********"" "**********"[ "**********"^ "**********"" "**********"] "**********"* "**********"" "**********"| "**********"\ "**********"' "**********"[ "**********"^ "**********"\ "**********"' "**********"] "**********"\ "**********"' "**********") "**********"$ "**********"' "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********") "**********"
        )

    def t_define(name: str, type: str, scope: dict[str, (str, int)], lable=None):
        assert type in ("variable", "function"), "E101"
        if type == "variable":
            global_data.append(0)
            scope[name] = (type, len(global_data) - 1)
        else:
            scope[name] = (type, lable)

    def set_varible(name: str, scope: dict[str, (str, int)], is_array=False) -> list[str]:
        return (
            [
                {"instruction": "LD", "operand": "SP+0"},
                {"instruction": "ST", "operand": "[" + str(scope[name][1]) + "]"},
            ]
            if is_array
            else [
                {"instruction": "LD", "operand": "SP+0"},
                {"instruction": "ST", "operand": str(scope[name][1])},
            ]
        )

    def compile_str(name: str, scope: dict[str, (str, int)]) -> list[str]:  # string, varible, number
        if t_is(name, "variable"):
            if name not in scope:
                print(name + " is undefined")
                exit(1)
            if scope[name][0] == "variable":
                return [
                    {"instruction": "LD", "operand": "[" + str(scope[name][1]) + "]"},
                    {"instruction": "PUSH"},
                ]
            print(name + " isn't variable")
            exit(1)
        if t_is(name, "string"):
            ind = len(global_data)
            global_data.extend([ord(c) for c in name[1:-1]])
            global_data.append(0)
            return [{"instruction": "LD", "operand": str(ind)}, {"instruction": "PUSH"}]
        if t_is(name, "number"):
            return [
                {"instruction": "LD", "operand": str(name)},
                {"instruction": "PUSH"},
            ]
        print("Unknown token: "**********"
        exit(1)

    def compile(ast: AST, scope: dict[str, (str, int)]):
        t_assert(len(ast.args) != 0, "Empty parentheses", ast)
        if isinstance(ast.args[0], AST) or t_is(ast.args[0], "string"):  # ((code) (code) (code) ...)
            for arg in ast.args[:-1]:
                if isinstance(arg, AST):
                    compile(arg, scope)
                    ast.code.extend(arg.code)
                    ast.code.append({"instruction": "POP"})
            if isinstance(ast.args[-1], AST):
                compile(ast.args[-1], scope)
                ast.code.extend(ast.args[-1].code)
            else:
                ast.code.extend(compile_str(ast.args[-1], scope))
        elif ast.args[0] in ("setq", "defvar", "setv"):
            t_assert(len(ast.args) == 3, "setq expects 2 arguments", ast)
            t_assert(
                not isinstance(ast.args[1], AST) and t_is(ast.args[1], "variable"),
                "setq expects a variable as the first argument",
                ast,
            )
            if ast.args[0] == "defvar":
                t_define(ast.args[1], "variable", scope)
            t_assert(
                ast.args[1] in scope and scope[ast.args[1]][0] == "variable",
                ast.args[1] + " is not variable",
                ast,
            )
            if isinstance(ast.args[2], AST):
                compile(ast.args[2], scope)
                ast.code.extend(ast.args[2].code)
            else:
                ast.code.extend(compile_str(ast.args[2], scope))
            ast.code.extend(set_varible(ast.args[1], scope, ast.args[0] == "setv"))
        elif ast.args[0] == "IN":
            t_assert(len(ast.args) == 1, ast.args[0] + " expects 0 arguments", ast)
            ast.code.extend([{"instruction": "IN"}, {"instruction": "PUSH"}])
        elif ast.args[0] == "compile-malloc":
            t_assert(len(ast.args) == 2, ast.args[0] + " expects 1 arguments", ast)
            t_assert(
                not isinstance(ast.args[1], AST) and t_is(ast.args[1], "number") and int(ast.args[1]) > 0,
                ast.args[0] + " expects a number as the first argument",
                ast,
            )
            ast.code.extend(
                [
                    {"instruction": "LD", "operand": str(len(global_data))},
                    {"instruction": "PUSH"},
                ]
            )
            for i in range(int(ast.args[1])):
                global_data.append(0)
        elif ast.args[0] in ("getv", "OUT"):
            t_assert(len(ast.args) == 2, ast.args[0] + " expects 1 argument", ast)
            if isinstance(ast.args[1], AST):
                compile(ast.args[1], scope)
                ast.code.extend(ast.args[1].code)
            else:
                ast.code.extend(compile_str(ast.args[1], scope))
            if ast.args[0] == "getv":
                ast.code.extend(
                    [
                        {"instruction": "LD", "operand": "[SP+0]"},
                        {"instruction": "ST", "operand": "SP+0"},
                    ]
                )
            else:
                ast.code.extend([{"instruction": "LD", "operand": "SP+0"}, {"instruction": "OUT"}])
        elif ast.args[0] in ("=", ">=", "!=", "+", "-", "*", "/", "%"):
            t_assert(len(ast.args) == 3, ast.args[0] + " expects 2 arguments", ast)
            for arg in ast.args[1:]:
                if isinstance(arg, AST):
                    compile(arg, scope)
                    ast.code.extend(arg.code)
                else:
                    ast.code.extend(compile_str(arg, scope))
            if ast.args[0] in ("+", "*"):
                ast.code.append({"instruction": "POP"})
                ast.code.append(
                    {
                        "instruction": ({"+": "ADD", "*": "MUL"}[ast.args[0]]),
                        "operand": "[SP+0]",
                    }
                )
                ast.code.append({"instruction": "ST", "operand": "SP+0"})
            if ast.args[0] in ("-", "/", "%"):
                ast.code.append({"instruction": "LD", "operand": "SP+1"})
                ast.code.append(
                    {
                        "instruction": ({"-": "SUB", "/": "DIV", "%": "MOD"}[ast.args[0]]),
                        "operand": "[SP+0]",
                    }
                )
                ast.code.append({"instruction": "ST", "operand": "SP+1"})
                ast.code.append({"instruction": "POP"})
            if ast.args[0] in ("=", ">=", "!="):
                lable1 = rnd_lable()
                lable2 = rnd_lable()
                ast.code.append({"instruction": "LD", "operand": "SP+1"})
                ast.code.append({"instruction": "CMP", "operand": "[SP+0]"})
                ast.code.append(
                    {
                        "instruction": ({"=": "JE", "!=": "JNE", ">=": "JGE"}[ast.args[0]]),
                        "V": lable1,
                    }
                )
                ast.code.append({"instruction": "LD", "operand": "0"})
                ast.code.append({"instruction": "JMP", "V": lable2})
                ast.code.append({"instruction": "LD", "operand": "1", "lable": lable1})
                ast.code.append({"instruction": "ST", "operand": "SP+1", "lable": lable2})
                ast.code.append({"instruction": "POP"})
        elif ast.args[0] == "defun":
            lable1 = rnd_lable()
            ast.code.append({"instruction": "JMP", "V": lable1})
            t_assert(
                len(ast.args) > 3,
                ast.args[0] + " expects more 3 arguments (name, arguments, ...body)",
                ast,
            )
            t_assert(
                not isinstance(ast.args[1], AST) and t_is(ast.args[1], "variable"),
                "defun expects a name as the first argument",
                ast,
            )
            t_assert(
                isinstance(ast.args[2], AST)
                and all([not isinstance(arg, AST) and t_is(arg, "variable") for arg in ast.args[2].args]),
                "defun expects a arguments as the second argument",
                ast,
            )
            t_assert(len(ast.args[2].args) > 0, "Еxpects one or more arguments", ast)
            fscope = dict(filter(lambda x: x[1][0] != "arg_variable", scope.items()))
            t_define(ast.args[1], "function", scope, ast.token_nuber)
            ast.code.append({"instruction": "**********": "lable_f" + str(ast.token_nuber)})
            for i in range(len(ast.args[2].args)):
                t_define(ast.args[2].args[i], "variable", fscope)
                ast.code.append(
                    {
                        "instruction": "LD",
                        "operand": "SP+" + str(len(ast.args[2].args) - i),
                    }
                )
                ast.code.append(
                    {
                        "instruction": "ST",
                        "operand": str(fscope[ast.args[2].args[i]][1]),
                    }
                )
            for arg in ast.args[3:-1]:
                if isinstance(arg, AST):
                    compile(arg, fscope)
                    ast.code.extend(arg.code)
                    ast.code.append({"instruction": "POP"})
            if isinstance(ast.args[-1], AST):
                compile(ast.args[-1], fscope)
                ast.code.extend(ast.args[-1].code)
            else:
                ast.code.extend(compile_str(ast.args[-1], fscope))
            ast.code.append({"instruction": "POP"})
            ast.code.append({"instruction": "ST", "operand": "SP+" + str(len(ast.args[2].args))})
            ast.code.append({"instruction": "RET"})
            ast.code.append({"instruction": "LD", "operand": "1", "lable": lable1})
            ast.code.append({"instruction": "PUSH"})
        elif ast.args[0] in ("while", "if"):
            t_assert(
                len(ast.args) > 2,
                ast.args[0] + " expects more 2 arguments (condition, ...body)",
                ast,
            )
            if ast.args[0] == "while":
                lable1 = rnd_lable()
                lable2 = rnd_lable()
                ast.code.append({"instruction": "NOP", "lable": lable1})
                if isinstance(ast.args[1], AST):
                    compile(ast.args[1], scope)
                    ast.code.extend(ast.args[1].code)
                else:
                    ast.code.extend(compile_str(ast.args[1], scope))
                ast.code.append({"instruction": "LD", "operand": "SP+0"})
                ast.code.append({"instruction": "CMP", "operand": "0"})
                ast.code.append({"instruction": "JE", "V": lable2})
                ast.code.append({"instruction": "POP"})
                for arg in ast.args[2:]:
                    if isinstance(arg, AST):
                        compile(arg, scope)
                        ast.code.extend(arg.code)
                        ast.code.append({"instruction": "POP"})
                ast.code.append({"instruction": "JMP", "V": lable1})
                ast.code.append({"instruction": "NOP", "lable": lable2})
            elif ast.args[0] == "if":
                lable1 = rnd_lable()
                if isinstance(ast.args[1], AST):
                    compile(ast.args[1], scope)
                    ast.code.extend(ast.args[1].code)
                else:
                    ast.code.extend(compile_str(ast.args[1], scope))
                ast.code.append({"instruction": "LD", "operand": "SP+0"})
                ast.code.append({"instruction": "CMP", "operand": "0"})
                ast.code.append({"instruction": "JE", "V": lable1})
                ast.code.append({"instruction": "POP"})
                for arg in ast.args[2:]:
                    if isinstance(arg, AST):
                        compile(arg, scope)
                        ast.code.extend(arg.code)
                ast.code.append({"instruction": "NOP", "lable": lable1})
        else:
            t_assert(ast.args[0] in scope, "Unknown token", ast)
            t_assert(
                scope[ast.args[0]][0] == "function",
                ast.args[0] + " is not function",
                ast,
            )
            for arg in ast.args[1:]:
                if isinstance(arg, AST):
                    compile(arg, scope)
                    ast.code.extend(arg.code)
                else:
                    ast.code.extend(compile_str(arg, scope))
            ast.code.append({"instruction": "CALL", "V": "lable_f" + str(scope[ast.args[0]][1])})
            for i in range(len(ast.args[1:]) - 1):
                ast.code.append({"instruction": "POP"})

    def link(asm: list[dict[str, str]]):
        labels = {}
        for i in range(len(asm)):
            instr = asm[i]
            if "lable" in instr:
                labels[instr["lable"]] = i
                del instr["lable"]
        for instr in asm:
            if ("V" in instr) and instr["V"] in labels:
                instr["V"] = labels[instr["V"]]
        return asm

    compile(ast, {})
    return [global_data] + link(ast.code + [{"instruction": "HALT"}])


"""
def print_ast(node: AST, depth=0):
    indent = '    ' * depth
    #print(f'{indent}Code: {";".join(node.code)}')
    print(f'{indent}Token nuber: "**********"
    print(f'{indent}Arguments:')
    for arg in node.args:
        if isinstance(arg, AST):
            print_ast(arg, depth + 1)
        else:
            print(f'{indent}  {arg}')
"""


def write_code(file_path, code):
    import json

    with open(file_path, "w") as file:
        json.dump(code, file)


def translate_code(source, target):
    with open(source, encoding="utf-8") as f:
        source = f.read()

    tokens = "**********"
    ast = "**********"
    asm = "**********"

    write_code(target, asm)
    print("source LoC:", len(source.split("\n")), "code instr:", len(asm))


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    translate_code(source, target)
