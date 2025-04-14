#date: 2025-04-14T17:07:22Z
#url: https://api.github.com/gists/7cff4d77a404ccdcf5a388ea1319ddde
#owner: https://api.github.com/users/teshanshanuka

# Simple brainfuck interpreter: https://en.wikipedia.org/wiki/Brainfuck
# Author: Teshan Liyanage <teshanuka@gmail.com>

# Use python3.10+

import sys


class BrainfuckInterpreter:
    def __init__(self, prog: str, input_="") -> None:
        self.prog = ""
        for c in prog:
            if c in "<>+-.,[]":
                self.prog += c

        self.input_ = [ord(c) for c in input_] + [-1]  # -1 for EOF
        self.pc = 0
        self.dp = 0  # data pointer
        self.ip = 0  # input pointer
        self.data = [0] * 30000  # should be at least of size 30000 accordig to specs

        brackets = self.find_bracket_pairs()
        # should have no overlap on keys and values so just use one dict
        self.jmp_table = {k: v for k, v in brackets} | {v: k for k, v in brackets}

    def find_bracket_pairs(self):
        stack, pairs = [], []
        for i, c in enumerate(self.prog):
            match c:
                case "[":
                    stack.append(i)
                case "]":
                    pairs.append((stack.pop(), i))
        assert len(stack) == 0, "Unbalanced bracket counts"
        return pairs

    def run(self):
        while self.pc < len(self.prog):
            jmped = False
            match self.prog[self.pc]:
                case ">":
                    self.dp += 1
                case "<":
                    self.dp -= 1
                case "+":
                    self.data[self.dp] += 1
                case "-":
                    self.data[self.dp] -= 1
                case ".":
                    print(chr(self.data[self.dp]), end="")
                case ",":
                    self.data[self.dp] = self.input_[self.ip]
                    self.ip += 1
                case "[":
                    if self.data[self.dp] == 0:
                        self.pc = self.jmp_table[self.pc]
                        jmped = True
                case "]":
                    if self.data[self.dp] != 0:
                        self.pc = self.jmp_table[self.pc]
                        jmped = True

            if not jmped:
                self.pc += 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"""Usage: python3 {sys.argv[0]} <code> [<input>]
Examples:
  0. Print 7 (loop example): ++>+++++[<+>-]++++++++[<++++++>-]<.
  1. Hello world: [.[.],..,,,+,-,<>,[]..]++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.
  2. ROT13: -,+[-[>>++++[>++++++++<-]<+<-[>+>+>-[>>>]<[[>+<-]>>+>]<<<<<-]]>>>[-]+>--[-[<->+++[-]]]<[++++++++++++<[>-[>+>>]>[+[<+>-]>+>>]<<<<<-]>>[<+>-]>[-[-<<[-]>>]<<[<<->>-]>>]<<[<<+>>-]]<[-]<.[-]<-,+]
""")
        sys.exit(1)

    interp = BrainfuckInterpreter(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    interp.run()
