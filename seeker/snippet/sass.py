#date: 2022-06-29T17:27:15Z
#url: https://api.github.com/gists/0d219f5db9da8bd7d2dd3e33d10bb8ff
#owner: https://api.github.com/users/CebolaBros64

sceneSet = set({})

with open("main.dol", 'rb') as f:
    bin = f.read()
    for i in range(0, len(bin)):
        try:
            if bin[i] == 0x00 and bin[i + 1] == 0x00 and bin[i + 2] == 0x09 and bin[i + 3] == 0x00 and bin[i + 4] == 0x00 and bin[i + 5] == 0x00 and bin[i + 6] == 0x00:
                # print(f"{i}: {hex(bin[i])} {hex(bin[i + 1])} {hex(bin[i + 2])} {hex(bin[i + 3])} {hex(bin[i + 4])} {hex(bin[i + 5])} {hex(bin[i + 6])} {hex(bin[i + 7])}")
                sceneSet.add(bin[i + 7])
        except IndexError:
            pass

for i in sceneSet:
    print(hex(i))