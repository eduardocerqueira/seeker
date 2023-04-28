#date: 2023-04-28T17:05:34Z
#url: https://api.github.com/gists/e0173b91a71b49279fc1806ab2851ad3
#owner: https://api.github.com/users/joel05reyes

words=input("Escribir palabras separadas por ,: ")
def agregar_ing(words:str)->str:
    separate= words.split(",")
    ordenated=[]
    for word in separate:
        reducir =word.strip()
        a=len(reducir)
        if a >= 3:
            char=reducir[-3:]
            if char == 'ing':
                reducir = reducir+'ly'
            else:
                reducir = reducir+'ing'
        else:
            reducir
        ordenated.append(word)
    ordenated.sort()
    return ",".join(ordenated)
print(agregar_ing(words))
