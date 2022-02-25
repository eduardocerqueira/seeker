#date: 2022-02-25T16:46:32Z
#url: https://api.github.com/gists/e8261866e636db6f8c006280524543c6
#owner: https://api.github.com/users/0788611515

def crop(message, k):
    if len(message) <= k:
       return message
    else:
       return ' '.join(message[:k+1].split(' ')[0:-1])
      
test1 = crop("Codibility We test coders", 14)
test2 = crop("The quick brown fox jumps over the lazy dog", 39)
test3 = crop("why not", 100)

print(test1)
print(test2)
print(test3)

# run it on this IDE online: https://extendsclass.com/python.html