#date: 2025-07-03T17:04:29Z
#url: https://api.github.com/gists/718d6c1f488e89601e45c0082c4e9500
#owner: https://api.github.com/users/endes0

import os

rom_file = open("rom.bin", "rb")
rom_data = rom_file.read()
rom_file.close()

n_of_files = rom_data[0x04:0x07]
n_of_files = int.from_bytes(n_of_files, byteorder='little')
print(f"Number of files: {n_of_files}")
for i in range(n_of_files):
    file_block_offset = 0x10 + i * 0x48
    file_name = rom_data[file_block_offset:file_block_offset + 0x40].decode('ascii').strip('\x00')
    file_start = int.from_bytes(rom_data[file_block_offset + 0x40:file_block_offset + 0x44], byteorder='little')
    file_size = int.from_bytes(rom_data[file_block_offset + 0x44:file_block_offset + 0x48], byteorder='little')
    file_data = rom_data[file_start:file_start + file_size]

    # Save the file, creating directories as needed
    if os.path.dirname(file_name) != "":
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as file_out:
        file_out.write(file_data)
    print(f"Extracted: {file_name} ({file_size} bytes)")