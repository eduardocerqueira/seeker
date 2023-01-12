#date: 2023-01-12T17:16:46Z
#url: https://api.github.com/gists/a4805b23384f7b70ff7b397e51110e67
#owner: https://api.github.com/users/linusrachlis

import traceback

indent = 0
label_stack = []


def stepin(label):
    global indent
    tmp_log(f"stepping into: {label}")
    label_stack.append(label)
    indent += 1


def stepout():
    global indent
    indent -= 1
    label = label_stack.pop()
    tmp_log(f"stepped out of: {label}")


def tmp_log(*args, tb=False):
    f = open("/tmp/debug.log", "a")
    for arg in args:
        to_write = str(arg)
        f.write((" " * indent) + to_write)
        f.write("\n")

    if tb:
        tb_list = traceback.format_stack()
        tb_str = "".join(tb_list)
        f.write(tb_str)
        f.write("\n")

    f.close()


def tmp_log_handle():
    f = open("/tmp/debug.log", "a")
    return f
