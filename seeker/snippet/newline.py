#date: 2023-06-14T16:56:57Z
#url: https://api.github.com/gists/6c147059fd32121e09e5eabc0b0ab219
#owner: https://api.github.com/users/VineethKumar7

import csv

# Assuming pcsi_list is a list of numbers

with open(txt_file_path_posi, 'w', newline='') as pcsi:
    csv_writer = csv.writer(pcsi)
    csv_writer.writerow(pcsi_list)  # Write the numbers as a single row

# Alternatively, write each number as a separate row
with open(txt_file_path_posi, 'w', newline='') as pcsi:
    csv_writer = csv.writer(pcsi)
    for number in pcsi_list:
        csv_writer.writerow([number])
