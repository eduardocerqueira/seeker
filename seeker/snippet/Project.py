#date: 2025-01-03T17:01:01Z
#url: https://api.github.com/gists/c3e5eafcb872db96470b916e55d93c02
#owner: https://api.github.com/users/sepehr21ar

class Stack:
    def __init__(self):
        self.items = []
        print("We will see stacks and output list first in each action, after that we solve postfix")

    def push(self, item):
        self.items.append(item)
        print(f"  Pushed: {item}")
        print(f"  Stack after push: {self.items}")

    def pop(self):
        if not self.is_empty():
            item = self.items.pop()
            print(f"  Popped: {item}")
            print(f"  Stack after pop: {self.items}")
            return item
        else:
            return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

    @staticmethod
    def infix_to_postfix(expression):
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        stack = Stack()
        output = []
        i = 0

        # حذف فاصله‌های اضافی
        expression = expression.replace(" ", "")

        print(f"Converting infix expression '{expression}' to postfix:")

        while i < len(expression):
            if expression[i].isdigit():
                num = ''
                while i < len(expression) and expression[i].isdigit():
                    num += expression[i]
                    i += 1
                output.append(num)
                print(f"  Found number: {num}")
                print(f"  Current Stack: {stack.items}")
                print(f"  Output so far: {output}")
            elif expression[i] in precedence:
                while stack.items and precedence.get(stack.peek(), 0) >= precedence.get(expression[i], 0):
                    output.append(stack.pop())
                stack.push(expression[i])
                i += 1
                print(f"  Operator: {expression[i - 1]}")
                print(f"  Current Stack: {stack.items}")
                print(f"  Output so far: {output}")
            elif expression[i] == '(':
                stack.push(expression[i])
                i += 1
                print(f"  Found '(': Pushed '(' onto the stack.")
                print(f"  Current Stack: {stack.items}")
                print(f"  Output so far: {output}")
            elif expression[i] == ')':
                while stack.items and stack.peek() != '(':
                    output.append(stack.pop())
                stack.pop()  # Remove '('
                i += 1
                print(f"  Found ')': Popped operators until '('.")
                print(f"  Current Stack: {stack.items}")
                print(f"  Output so far: {output}")

        while stack.items:
            output.append(stack.pop())

        print(f"Final Postfix Output: {output}")
        return ' '.join(output)

    @staticmethod
    def solve_postfix(expression):
        print("\n\n")
        stack = Stack()

        # Define operator functions
        def add(x, y):
            return x + y

        def subtract(x, y):
            return x - y

        def multiply(x, y):
            return x * y

        def divide(x, y):
            return x / y

        def power(x, y):
            return x ** y

        operators = {'+': add, '-': subtract, '*': multiply, '/': divide, '^': power}

        print("\nSolving Postfix expression:")

        for char in expression.split():
            if char.isdigit():
                stack.push(int(char))
            elif char in operators:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operators[char](operand1, operand2)
                stack.push(result)
                print(f"  Applied operator {char}: {operand1} {char} {operand2} = {result}")
                print(f"  Stack after operation: {stack.items}")

        result = stack.pop()
        print(f"\nFinal Result: {result}")
        return result


# دریافت ورودی
infix_expression = input("Enter an expression (with spaces between numbers and operators): ")
postfix_expression = Stack.infix_to_postfix(infix_expression)
result = Stack.solve_postfix(postfix_expression)

# نمایش نتایج
print(f"\nPostfix: {postfix_expression}")
print(f"Result: {result}")
