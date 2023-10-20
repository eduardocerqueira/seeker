#date: 2023-10-20T17:09:22Z
#url: https://api.github.com/gists/c7502eca66308515d8fb3b32d39f1d54
#owner: https://api.github.com/users/mypy-play

height = 8
print('\n'.join(f'''{f'{"  ":#^{wall}}':>{lpad}}''' for (lpad, wall) in enumerate(range(4, height * 2 + 3, 2), height + 3)))