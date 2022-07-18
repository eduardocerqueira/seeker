#date: 2022-07-18T17:11:03Z
#url: https://api.github.com/gists/8886f9926c06ade158102b70fd40b8c6
#owner: https://api.github.com/users/chengscott

import argparse
import json
import math
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename')
  parser.add_argument('-o', '--output-dir')
  parser.add_argument('-n', type=int)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  with open(args.filename) as f:
    j = json.load(f)
    jo = dict(schemaVersion=j['schemaVersion'])
    iters = math.ceil(len(j['traceEvents']) / args.n)
    for i in range(iters):
      name = f'{args.output_dir}/{os.path.basename(args.filename)[:-4]}{i}.json'
      ji = dict(jo)
      ji['traceName'] = name
      ji['traceEvents'] = j['traceEvents'][i * args.n:(i + 1) * args.n]
      with open(name, 'w') as fi:
        json.dump(ji, fi)

if __name__ == '__main__':
    main()