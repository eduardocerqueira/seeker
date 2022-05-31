#date: 2022-05-31T17:20:40Z
#url: https://api.github.com/gists/55bcb9072ae7c0ba869b242d15c9d1e0
#owner: https://api.github.com/users/astherath

import sys

DEFAULT_INPUT_FILENAME = "stmt.csv"
DEFAULT_OUTPUT_FILENAME = "out_stmt.csv"
OUTPUT_FILENAME = None
INPUT_FILENAME = None
AVG_INPUT_FILENAME = None
AVG_OUTPUT_FILENAME = None


def get_file_data(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    return lines


def write_file_data(filename, lines):
    with open(filename, "w") as f:
        f.writelines(lines)
    print(f"wrote {len(lines)} to {filename}")


def from_lines_to_data_tuples(lines):
    description_index = 1
    amount_index = 2

    data = []

    lines = lines[8:]  # skipping data

    for line in lines:
        cells = [x.replace(",", "").replace('"', "") for x in line.split(",")]
        description = cells[description_index]
        try:
            amount = str(abs(float(cells[amount_index])))
        except Exception as e:
            print(f"couldnt cast line: {line}, ex: {e}")
            raise e

        if (new_desc := is_description_cc(description)):
            transaction_data = (new_desc, amount)
            data.append(transaction_data)

    print(f"lines read: {len(lines)}, lines accepted: {len(data)}")

    return data


def is_description_cc(desc):
    cc_keywords = ["AMERICAN EXPRESS", "DISCOVER", "APPLECARD"]
    for word in cc_keywords:
        if word in desc:
            return word
    return None


def write_data_tuples_to_csv_output(data):
    csv_lines = ["description,amount\n"]  # header
    data_lines = [f"{','.join(x)}\n" for x in data]

    csv_lines.extend(data_lines)

    write_file_data(OUTPUT_FILENAME, csv_lines)

    print(f"wrote {len(csv_lines) - 1} lines to output...")


def get_avg_for_each_card():
    data = get_file_data(AVG_INPUT_FILENAME)[1:]  # skip header
    hash_set = {}
    for line in data:
        points = line.split(",")
        desc = points[0]
        amount = float(points[1])

        try:
            hash_set[desc].append(amount)
        except:
            hash_set[desc] = [amount]

    return [(k, (sum(v) / len(v))) for k, v in hash_set.items()]


def write_avg_data_to_csv(data):
    output = ["card name,avg\n"]
    output.extend([f"{','.join([a, str(b)])}\n" for a, b in data])

    write_file_data(AVG_OUTPUT_FILENAME, output)


def SET_GLOBALS():
    args = sys.argv[1:]
    args.extend([None for x in range(10)])
    global INPUT_FILENAME, OUTPUT_FILENAME, AVG_INPUT_FILENAME, AVG_OUTPUT_FILENAME
    INPUT_FILENAME = args[0] if args[0] else DEFAULT_INPUT_FILENAME
    OUTPUT_FILENAME = args[1] if args[1] else DEFAULT_OUTPUT_FILENAME

    AVG_INPUT_FILENAME = OUTPUT_FILENAME
    AVG_OUTPUT_FILENAME = f"processed_{INPUT_FILENAME}"

    print(f"using input: {INPUT_FILENAME} and output: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    SET_GLOBALS()

    data = from_lines_to_data_tuples(get_file_data(INPUT_FILENAME))
    write_data_tuples_to_csv_output(data)

    avg_data = get_avg_for_each_card()
    write_avg_data_to_csv(avg_data)
