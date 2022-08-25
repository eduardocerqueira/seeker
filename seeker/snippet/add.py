#date: 2022-08-25T17:04:30Z
#url: https://api.github.com/gists/7fb96778336fea5b443b92c5b05f330c
#owner: https://api.github.com/users/mondrasovic

import sys

def main():
  x = int(sys.argv[1])
  y = int(sys.argv[2])
  
  res = x + y
  
  print(f"{x} + {y} = {res})
  
  return 0

if __name__ == '__main__':
  sys.exit(main())
