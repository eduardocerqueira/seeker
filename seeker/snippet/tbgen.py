#date: 2024-06-20T16:48:18Z
#url: https://api.github.com/gists/33c59f2a3e9a60b6336af0de982df940
#owner: https://api.github.com/users/ronejfourn

#!/bin/python3
import sys, re

def fatal(r):
    print("tbgen: fatal:", r)
    exit(1)

def warn(r):
    print("tbgen: warning:", r)

if len(sys.argv) < 2:
    fatal("no input file")

try:
    with open(sys.argv[1], "r") as fp:
        fd = fp.read()
except Exception as e:
    fatal("input file error : " + str(e))

fd = re.sub(r"//[^\n^\r]*", '', fd)
fd = re.sub(r"`[^\n^\r]*", '', fd)
fd = re.sub(r"/\*.*\*/", '', fd)

if "module" not in fd:
    fatal("no module found")

if "endmodule" not in fd:
    fatal("endmodule not found")

fd = fd[fd.index("module") + 6:fd.index("endmodule")].strip()
mn = fd[:re.search("\(|;", fd).start()].strip()
os = "module " + mn + "_tb;\n\n// inputs\n"

hc = False
ii = [m.end() for m in re.finditer("(?<![_a-zA-Z0-9])input(?![_a-zA-Z0-9])", fd)]
if ii:
    for i in ii:
        p = re.search(";|\)|,\s*input|,\s*output", fd[i:]).start()
        p = fd[i:i + p].strip()
        if p.startswith("reg"):
            p = p[3:]
        elif p.startswith("wire"):
            p = p[4:]
        p = "".join(p.split()).replace(",", ", ").replace("]", "] ")
        p = "reg " + p + ";"
        os += p + "\n"
        hc = hc or "clk" in p
else:
    warn("no inputs found")

os += "\n// outputs\n"

ii = [m.end() for m in re.finditer("(?<![_a-zA-Z0-9])output(?![_a-zA-Z0-9])", fd)]
if ii:
    for i in ii:
        p = re.search(";|\)|,\s*input|,\s*output", fd[i:]).start()
        p = fd[i:i + p].strip()
        if p.startswith("reg"):
            p = p[3:]
        elif p.startswith("wire"):
            p = p[4:]
        p = "".join(p.split()).replace(",", ", ").replace("]", "] ")
        p = "wire " + p + ";"
        os += p + "\n"
else:
    warn("no outputs found")

os += "\n//clock period\nparameter PERIOD = 10;\n" if hc else "";
os += "\n// instantiate dut\n"
os += mn + " dut(.*);\n"
os += "\ninitial\nbegin\n\t$dumpvars(0, dut);\n"
os += "\tclk = 1'b0; #(PERIOD/2);\n\tforever\n\t\t#(PERIOD/2) clk = ~clk;\n" if hc else ""
os += "end\n\nendmodule"

try:
    with open(mn + "_tb.v", "w") as of:
        of.write(os)
    print("tbgen: output written to " + mn + "_tb.v")
except Exception as e:
    fatal("output file error : " + str(e))