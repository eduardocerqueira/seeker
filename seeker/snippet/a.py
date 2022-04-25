#date: 2022-04-25T17:08:11Z
#url: https://api.github.com/gists/afd774be64af445f1c2635161bb162f9
#owner: https://api.github.com/users/GusainMayank

# For file input use terminal pipes
# Like if I/P file name is in, this code script file name is a
# Then type
# python a.py < in
# May need to use python3 instead
# Advantage: no changes in code from stdin and file I/O

from sys import stdin

def main(): # O(#lines * |word|) Time, O(|word|) Space
    for line in stdin:
        word1, word2 = line.split()
        linearDistance, circularDistance = 0, 0
        for char1, char2 in zip(word1, word2):
            currentLinearDistance = abs(ord(char1) - ord(char2))
            linearDistance += currentLinearDistance
            circularDistance += min(currentLinearDistance, 26 - currentLinearDistance)
        
        print(f'{linearDistance} {circularDistance}')

if __name__ == '__main__':
    main()