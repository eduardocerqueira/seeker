#date: 2022-12-16T17:09:10Z
#url: https://api.github.com/gists/7884d6a69ef6334e9ce38eb8bd93d49d
#owner: https://api.github.com/users/JoelAtDeluxe

def decode_bin(s):
    chars = int(len(s)/8)
    for i in range(chars):
        ch = s[i*8 : (i+1)*8]
        idx = int(ch, 2)
        print(chr(idx), end='')
    print()
