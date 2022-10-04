#date: 2022-10-04T17:23:59Z
#url: https://api.github.com/gists/9e8f298b5fcf005f5f15296964a6c8b1
#owner: https://api.github.com/users/rejgan318

def dataclass2csv(filename: str, rows: list[Item], header: bool = True):
    with open(filename, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if header:
            csv_writer.writerow(['group', 'subgroup', 'name'])
        csv_writer.writerows([astuple(row) for row in rows])