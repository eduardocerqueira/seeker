#date: 2024-06-26T16:41:31Z
#url: https://api.github.com/gists/de7624d7f5fbf1714792caf9d8184681
#owner: https://api.github.com/users/OlivierKobialka

import tkinter as tk


def encrypt(value: str, depth: int) -> str:
    # Create a 2D list to represent the rail fence pattern
    rail = [['\n' for _ in range(len(value))] for _ in range(depth)]

    direction = False
    row, col = 0, 0

    for char in range(len(value)):
        if (row == 0) or (row == depth - 1):
            direction = not direction

        rail[row][col] = value[char]
        col += 1

        if direction:
            row += 1
        else:
            row -= 1

    result = []

    for d in range(depth):
        for c in range(len(value)):
            if rail[d][c] != '\n':
                result.append(rail[d][c])

    return "".join(result)


def decrypt(value: str, depth: int) -> str:
    # Create a 2D list to represent the rail fence pattern
    rail = [['\n' for _ in range(len(value))] for _ in range(depth)]

    dir_down = None
    row, col = 0, 0

    for i in range(len(value)):
        if row == 0:
            dir_down = True
        if row == depth - 1:
            dir_down = False

        rail[row][col] = '*'
        col += 1

        if dir_down:
            row += 1
        else:
            row -= 1

    index = 0

    for i in range(depth):
        for char in range(len(value)):
            if ((rail[i][char] == '*') and
                    (index < len(value))):
                rail[i][char] = value[index]
                index += 1

    result = []
    row, col = 0, 0

    for i in range(len(value)):
        if row == 0:
            dir_down = True
        if row == depth - 1:
            dir_down = False

        if rail[row][col] != '*':
            result.append(rail[row][col])
            col += 1

        if dir_down:
            row += 1
        else:
            row -= 1
    return "".join(result)


def encrypt_button_clicked():
    input_text = input_entry.get()
    depth = int(depth_entry.get())
    result = encrypt(input_text, depth)
    result_label.config(text=f"Encrypted: {result}")


def decrypt_button_clicked():
    input_text = input_entry.get()
    depth = int(depth_entry.get())
    result = decrypt(input_text, depth)
    result_label.config(text=f"Decrypted: {result}")


window = tk.Tk()
window.title("Rail Fence Cipher")
input_label = tk.Label(window, text="Input Text:")
input_label.pack()

input_entry = tk.Entry(window)
input_entry.pack()

depth_label = tk.Label(window, text="Depth:")
depth_label.pack()

depth_entry = tk.Entry(window)
depth_entry.pack()

encrypt_button = tk.Button(window, text="Encrypt", command=encrypt_button_clicked)
encrypt_button.pack()

decrypt_button = tk.Button(window, text="Decrypt", command=decrypt_button_clicked)
decrypt_button.pack()

result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()