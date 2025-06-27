#date: 2025-06-27T17:11:58Z
#url: https://api.github.com/gists/5447e6b2d07cbb16adb5d52f225022e3
#owner: https://api.github.com/users/mypy-play

byt = b'abc'
st = 'hello'
btar = bytearray()
num = 1

byt == st  # E: Non-overlapping equality check (left operand type: "bytes", right operand type: "str")
byt == num  # E: Non-overlapping equality check (left operand type: "bytes", right operand type: "int")
btar == num  # E: Non-overlapping equality check (left operand type: "bytearray", right operand type: "int")
