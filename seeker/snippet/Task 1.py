#date: 2025-11-05T17:02:40Z
#url: https://api.github.com/gists/5c214977bc22ba98c87bbbb56db89ca5
#owner: https://api.github.com/users/Joycarey

x =int(input("Enter the First number:"))
y =int(input("Enter the Second number:"))
  
add = x + y 
sub = x - y 
pro = x * y 
div = x / y 
  
print("Addition:",add) 
print("Subtraction :",sub) 
print("Multiplication:",pro) 
print("Division:",div)

# Initializing the value of a, b, and c
a = int(input("Enter the First number: "))
b = int(input("Enter the Second number: "))
c = int(input("Enter the Third number: "))

# Using relational operators
print(a, ">", b, "is", a > b)
print(a, "<", b, "is", a < b)
print(c, "==", a, "is", c == a)
print(c, "!=", b, "is", c != b)
print(a, ">=", b, "is", a >= b)
print(b, "<=", a, "is", b <= a)

# Taking three numbers as input
a = int(input("Enter the First number: "))
b = int(input("Enter the Second number: "))
c = int(input("Enter the Third number: "))

# Performing logical operations
print("\nLogical Operations Results:")
print((a > b) and (b > c))
print((a > b) or (b > c))
print(not(a > b))
print(not(b > c))