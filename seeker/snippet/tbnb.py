#date: 2024-05-30T16:48:04Z
#url: https://api.github.com/gists/de3f18c7d9dad7ee662d3dac207eea39
#owner: https://api.github.com/users/Gilbert189

import sys

def tbnb(input):
  memory = "".join(f"{x:08b}" for x in input)
  memory = [memory[i:i+2] for i in range(0, len(memory), 2)]
  ip = 0
  def inc():
    nonlocal ip
    ip += 1
    ip %= len(memory)

  while True:
    if memory[ip] == "01":  # ACT
      inc()
      pointed = memory[(ip + int(memory[ip], 2)) % len(memory)]
      memory[(ip + int(memory[ip], 2)) % len(memory)] = {
        "00": "11",
        "01": "10",
        "10": "00",
        "11": "01",
      }[pointed]
    elif memory[ip] == "10":  # JMP
      inc()
      ip += int(memory[ip], 2)
    elif memory[ip] == "11":  # END
      break
    # NOP does nothing.
    inc()
  memory = "".join(memory)
  memory = bytes(int(memory[i:i+8], 2) for i in range(0, len(memory), 8))
  return memory

if __name__ == "__main__":
  result = tbnb(sys.stdin.buffer.read())
  sys.stdout.buffer.write(result)