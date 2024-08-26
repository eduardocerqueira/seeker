#date: 2024-08-26T16:52:15Z
#url: https://api.github.com/gists/963fe34822e3b61bf6e2c7bd54e86ea7
#owner: https://api.github.com/users/colinbr96

#! /usr/local/bin/python3
import argparse
import sys
import textwrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Column width to wrap on (default 90)",
        default=90,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force long lines to wrap, sometimes breaking words",
        default=False,
    )
    parser.add_argument("-o", "--output", help="Output filename to write to")
    return parser.parse_args()


def main():
    args = parse_args()

    content = None
    with open(args.filename) as f:
        content = f.readlines()

    output = "\n".join(
        [
            "\n".join(
                textwrap.wrap(
                    line,
                    args.width,
                    break_long_words=args.force,
                    replace_whitespace=False,
                )
            )
            for line in content
            if line.strip()
        ]
    )

    if args.output:
        print(f"Writing to {args.output}")
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
