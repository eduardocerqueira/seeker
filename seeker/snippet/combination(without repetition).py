#date: 2022-02-21T17:11:17Z
#url: https://api.github.com/gists/98e948aa0131f1dc956079fbbb79a6b5
#owner: https://api.github.com/users/Ahmedsaed

# Source: https://stackoverflow.com/questions/127704/algorithm-to-return-all-combinations-of-k-elements-from-n
def main():
    for i in list(choose_iter(['a', 'b', 'c'],2)):
        print(i)

def choose_iter(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield [elements[i]]
        else:
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield [elements[i],] + next

main()