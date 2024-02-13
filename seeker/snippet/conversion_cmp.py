#date: 2024-02-13T17:02:15Z
#url: https://api.github.com/gists/0bbbcdd253816d0e2e19ce0c226aedfd
#owner: https://api.github.com/users/ankittare

import csv

def compare_csv(file1, file2, output_file, exclude_column):
  """
  Compares two CSV files and outputs lines with differences, excluding a specified column.

  Args:
    file1: Path to the first CSV file.
    file2: Path to the second CSV file.
    output_file: Path to the output file where differences will be written.
    exclude_column: Name of the column to exclude from comparison.
  """
  with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(out)

    headers = next(reader1)  # Read and skip headers
    exclude_index = headers.index(exclude_column)  # Get index of the excluded column

    for row1, row2 in zip(reader1, reader2):
      if row1[exclude_index] != row2[exclude_index]:
        continue  # Skip if only excluded column differs

      if row1 != row2:
        writer.writerow(f"Difference in line {reader1.line_num}:")
        writer.writerow(["Column", "File 1", "File 2"])
        for i, (val1, val2) in enumerate(zip(row1, row2)):
          if i != exclude_index and val1 != val2:
            writer.writerow([i + 1, val1, val2])

if __name__ == "__main__":
  # Replace these with your actual file paths
  file1 = "final_results_with_transform.csv"
  file2 = "final_results_without_transform.csv"
  output_file = "differences.csv"
  exclude_column = "batch_load_datetime" # exclude this column since it depends on current date time. 

  compare_csv(file1, file2, output_file, exclude_column)
  print(f"Differences written to: {output_file}")