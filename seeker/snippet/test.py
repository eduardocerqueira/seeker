#date: 2024-12-23T16:53:47Z
#url: https://api.github.com/gists/ccc26269b5442f8c6005c2916a221f07
#owner: https://api.github.com/users/AND3SIL4

# 1. Test one
# List of words
lista: list[str] = ["Hola", "hola", "perro", "gato"]
word: str = "hola"
lower_list: list[str] = [element.lower() for element in lista]

print("Amount of values in list:", lower_list.count(word))

# 2. Test two
# Set any matrix (Ex: 3x3)
matriz = [
    [10, 2, 3],
    [4, 12, 6],
    [7, 8, 9]
]

# Get the less number in only one line
less_number: int = min(min(fila) for fila in matriz)
print("The less value in the matrix is:", less_number)

# Get the biggest number in only one line
biggest_number: int = max((max(fila) for fila in matriz))
print("The biggest number is:", biggest_number)