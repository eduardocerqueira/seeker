#date: 2021-11-09T17:04:28Z
#url: https://api.github.com/gists/9e47a4d728d0834d660301abe2017cd1
#owner: https://api.github.com/users/FawnKitten


# rpc.py
# a calculator for reverse polish notation

# reverse polish notation
# syntax:
#   operator : '+' | '-' | '*' | '/'
#   expr : NUMBER | expr expr operator

OPERATORS = ['+', '-', '*', '/']

def isoperator(token):
    return token in OPERATORS

def apply_opperator(operands, operator):
    assert (len(OPERATORS) == 4) , "Non Exhaustive Handling"
    if operator == '+':   return operands[0] + operands[1]
    elif operator == '-': return operands[0] - operands[1]
    elif operator == '/': return operands[0] / operands[1]
    elif operator == '*': return operands[0] * operands[1]

def expr(tokens):
    nums = []
    res = 0
    for token in tokens:
        if token.isdigit():
            nums.append(int(token))
        elif isoperator(token):
            val1 = nums.pop()
            val2 = nums.pop()
            nums.append(apply_opperator((val1, val2), token))
        else:
            raise SyntaxError(f"Error at token {token!r}")
    if len(nums) != 1: raise SyntaxError("Invalid Syntax")
    return nums[0]

def calculate(text):
    tokens = text.split()
    try:
        num = expr(tokens)
        return num
    except SyntaxError as e:
        print(e)
        return None

def main():
    while True:
        try:
            text = input("> ")
            if text:
                print(calculate(text) or '')
        except EOFError:
            print("Bye.")
            break

if __name__ == "__main__":
    main()
