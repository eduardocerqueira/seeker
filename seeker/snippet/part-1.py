#date: 2023-03-07T16:51:29Z
#url: https://api.github.com/gists/f8fcdfb3f0bae67eabb5726947268e24
#owner: https://api.github.com/users/lioraggold

#PART A
file = open("u2t1-2023-sotu.txt", "r")
word_counts = {}
for line in file:
    for word in line.split():
        if word in word_counts:
            word_counts[word] = word_counts[word] + 1
        else:
            word_counts[word] = 1

print( "There were " + str( len(word_counts) ) + " total words in this file.")   



