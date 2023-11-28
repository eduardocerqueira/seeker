#date: 2023-11-28T16:45:41Z
#url: https://api.github.com/gists/40454798ae53386a1d5b9c8bb60664d5
#owner: https://api.github.com/users/Lorenzoantonelli

import sys
import os


def report_to_latex(report):
    if report[0] == '\n':
        report = report[1:]
    if report[-1] == '\n':
        report = report[:-1]

    lines = report.split('\n')

    header = ["\\begin{table}",
              "\\caption{Latex Table from Classification Report}",
              "\\label{table:classification:report}",
              "\\centering",
              "\\begin{tabular}{c c c c r}",
              "& Precision & Recall & F-score & Support",
              "\\\\"]

    body = []
    for line in lines[2:-4]:
        row = line.split()
        if len(row) == 5:
            body.append(" & ".join(row) + "\\\\")

    body.append("\\\\")

    footer = []
    for line in lines[-3:]:
        row = line.split()
        if len(row) == 3:
            footer.append("{} & & & {} & {}\\\\".format(*row))
        elif len(row) == 6:
            footer.append("{} {} & {} & {} & {} & {}\\\\".format(*row))

    footer.extend(["\\end{tabular}", "\\end{table}"])

    latex_table = '\n'.join(header + body + footer)

    return latex_table


def print_usage():
    fname = os.path.basename(__file__)
    print(f"Usage: python {fname} <report.txt>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    report = sys.argv[1]
    with open(report, 'r') as f:
        report = f.read()

    latex_table = report_to_latex(report)
    print(latex_table)
