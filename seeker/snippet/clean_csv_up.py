#date: 2024-11-08T16:49:15Z
#url: https://api.github.com/gists/dfc1ce15a4aae3ae7b91bad1621032d6
#owner: https://api.github.com/users/kaitouctr

import argparse
import csv
import itertools
from pathlib import Path


def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def get_base_name(name_list: list[str]):
    names_split = [name.split(" ") for name in name_list]
    base_name = ""
    for word_index in range(len(min(names_split, key=len))):
        name_str = " ".join(names_split[0][0:word_index+1])
        if all_equal([" ".join(name[0:word_index+1]) for name in names_split]):
            base_name = name_str
    return base_name


def clean_up_csv(filepath: Path):
    fieldnames = (
        "Number",
        "Name",
        "Form",
        "HP",
        "Attack",
        "Defense",
        "Sp.Attack",
        "Sp.Defense",
        "Speed"
    )
    with filepath.open("r", encoding="utf-8-sig",
        newline="") as f:
        c_reader = csv.DictReader(f, fieldnames=fieldnames)
        next(c_reader, None)
        csv_data = [row for row in c_reader]
    seen_numbers: set[str] = set()
    duplicate_numbers: set[str] = set()
    for row in csv_data:
        if (row["Number"]) not in seen_numbers:
            seen_numbers.add(row["Number"])
        else:
            duplicate_numbers.add(row["Number"])
    duplicate_rows = tuple([
        # This strange list comprehension allows for better typehinting
        tuple([x for x in enumerate(csv_data) if x[1]["Number"] == num])
        for num in duplicate_numbers
    ])
    for duplicate_row in duplicate_rows:
        pkmn_name = get_base_name([altform[1]["Name"] for altform in duplicate_row])
        for altform in duplicate_row:
            altform_name = altform[1]["Name"].removeprefix(pkmn_name).strip()
            altform[1]["Name"] = pkmn_name
            altform[1]["Form"] = altform_name
    modified_csv_data = tuple([altform for pkmn in duplicate_rows for altform in pkmn])
    for mod_row in modified_csv_data:
        csv_data[mod_row[0]] = mod_row[1]
    fixed_csv_path = filepath.resolve().with_stem(filepath.stem + "-fixed")
    with fixed_csv_path.open("x", encoding="utf-8", newline="") as f:
        c_writer = csv.DictWriter(f, fieldnames=fieldnames)
        c_writer.writeheader()
        for row in csv_data:
            c_writer.writerow(row)


def main():
    arg_parser = argparse.ArgumentParser(
        prog="Scraped CSV Cleaner",
        description="Cleans up CSVs made with my other Python script",
    )
    arg_parser.add_argument("filepath")
    args = arg_parser.parse_args()
    clean_up_csv(Path(args.filepath))
    


if __name__ == "__main__":
    main()