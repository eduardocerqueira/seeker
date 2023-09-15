#date: 2023-09-15T17:06:45Z
#url: https://api.github.com/gists/4a550107957698ab7af03dd10b4e664e
#owner: https://api.github.com/users/tuhdo

import csv

# Define the range of numbers for operands and the file name

# Create and open the CSV file for writing
def gen_dataset(fname, op):
    start = 1
    end = 500

    with open(fname, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write a header row (optional)
        csv_writer.writerow(['Operand1', 'Operand2', 'Sum'])

        # Generate and write the addition dataset
        if "+" == op:
            for operand1 in range(start, end + 1):
                for operand2 in range(start, end + 1):
                    result = operand1 + operand2
                    csv_writer.writerow([operand1, operand2, result])
        elif "*" == op:
            for operand1 in range(start, end + 1):
                for operand2 in range(start, end + 1):
                    result = operand1 * operand2
                    csv_writer.writerow([operand1, operand2, result])

        print(f"Arithmetic {op} dataset generated and saved to {fname}.")

gen_dataset("add_dataset.csv", "+")
gen_dataset("mul_dataset.csv", "*")
