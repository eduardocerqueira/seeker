#date: 2024-03-25T16:54:19Z
#url: https://api.github.com/gists/055745d216edd965ff2631db5ad94714
#owner: https://api.github.com/users/tos-kamiya

import argparse
import ast
import sys

def get_variables(expr):
    tree = ast.parse(expr, mode='eval')
    variables = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    return sorted(set(variables))

def evaluate_expressions(outp, expressions, inps, input_file_names = None):
    num_files = len(inps)
    vars = [chr(97 + i) for i in range(num_files)]
    pred_funcs = [eval(f"lambda {', '.join(vars)}: " + e) for e in expressions]

    current_items = [inp.readline().strip() for inp in inps]

    min_item = min(filter(None, current_items), default=None)
    while min_item is not None:
        vars_values = [(1 if ci == min_item else 0) for ci in current_items]
        values = [pf(*vars_values) > 0 for pf in pred_funcs]
        if any(values):
            print("\t".join((min_item if v else "") for v in values), file=outp)

        for i, inp in enumerate(inps):
            ci = current_items[i]
            if ci is None:
                continue
            if ci == min_item:
                next_item = inp.readline().strip()
                if next_item and next_item <= ci:
                    if input_file_names:
                        print(f"Error: File {input_file_names[i]} is not sorted", file=sys.stderr)
                    else:
                        print(f"Error: {i + 1}th file is not sorted", file=sys.stderr)
                    sys.exit(1)
                current_items[i] = next_item

        min_item = min(filter(None, current_items), default=None)

def main():
    parser = argparse.ArgumentParser(description="flexcomm - set operations on sorted files")
    parser.add_argument("files", nargs="+", help="input files")
    parser.add_argument("-p", "--predicate", action="append", required=True,
                        help="predicate expression (e.g., 'a - b')")

    args = parser.parse_args()
    if len(args.files) > 26:
        print(f"Error: Too many files.", file=sys.stderr)
        sys.exit(1)

    vars = [chr(97 + i) for i, _ in enumerate(args.files)]
    for p in args.predicate:
        pred_vars = get_variables(p)
        unknown_vars = sorted(set(pred_vars).difference(vars))
        if unknown_vars:
            print(f"Error: Invalid variable name(s) in expression: {', '.join(unknown_vars)}", file=sys.stderr)
            exit(1)

    inps = [open(f) for f in args.files]
    evaluate_expressions(sys.stdout, args.predicate, inps, input_file_names = args.files)
    for inp in inps:
        inp.close()

if __name__ == "__main__":
    main()
