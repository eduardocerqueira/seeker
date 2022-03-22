#date: 2022-03-22T16:57:44Z
#url: https://api.github.com/gists/e2620eeb0dd91ffa59ae785146ddc044
#owner: https://api.github.com/users/bmoreau

import os                                                                                                                                       from pathlib import Path
import csv

def validate(path_to_file) -> bool:
    is_valid = True
    with open(path_to_file, newline='') as bar_file:
        bar_reader = csv.DictReader(bar_file)
        for row in bar_reader:
            if int(row['NbWaiters']) == 0:
                print('Bar ', row['id'], ' does not have any waiter!')
                is_valid = False
            elif int(row['RestockQty']) == 0:
                print('Bar ', row['id'], ' cannot restock!')
                is_valid = False
    return is_valid

def main():
    data_path = Path(os.environ["CSM_DATASET_ABSOLUTE_PATH"])
    bar_path = data_path / "Bar.csv"
    test_passed = validate(bar_path)
    if not test_passed:
        print("Dataset validation failed.")
        exit(1)

if __name__ == "__main__":
    main()