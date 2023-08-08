#date: 2023-08-08T17:03:31Z
#url: https://api.github.com/gists/1c51d2b32e4501d6bb906585dfafa320
#owner: https://api.github.com/users/uruskan

def FirstReverse(strParam):
    ay = []  # Create an empty array
    for char in strParam:
        ay.append(char)  # Add each character to the array
    
    reversed_str = ""  # Initialize an empty string to store the reversed string
    
    i = len(ay) - 1
    while i >= 0:
        reversed_str += ay[i]  # Concatenate each character to the reversed string
        i -= 1
    
    return reversed_str

# Keep this function call here
print(FirstReverse(input()))
