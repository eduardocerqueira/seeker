#date: 2025-06-27T16:45:41Z
#url: https://api.github.com/gists/f19e381f74746a432e623d438e0bd87b
#owner: https://api.github.com/users/iwashi623

DEFAULT_WIDTH = 6

def nl(file_input=None, start_number=1, number_blank_lines=True):
    try:
        if file_input:
            file_obj = open(file_input, 'r')
        else:
            file_obj = sys.stdin
        
        try:
            line_number = start_number
            for line in file_obj:
                if number_blank_lines or line.strip():
                    print(f"{line_number:{DEFAULT_WIDTH}}\t{line}", end='')
                    line_number += 1
                else:
                    print(line, end='')
        finally:
            if file_input:
                file_obj.close()
            
    except FileNotFoundError:
        print(f"nl: {file_input}: No such file or directory", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"nl: {file_input}: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='number lines of files')
    parser.add_argument('file', nargs='?', help='file to process (default: stdin)')
    parser.add_argument('-v', '--starting-line-number', type=int, default=1,
                        help='start number (default: 1)')
    parser.add_argument('-b', '--body-numbering', type=str, choices=['a', 't'], default='a',
                        help='body numbering: a=all lines, t=text lines only (default: a)')
    
    args = parser.parse_args()
    number_blank_lines = (args.body_numbering == 'a')
    nl(args.file, args.starting_line_number, number_blank_lines)