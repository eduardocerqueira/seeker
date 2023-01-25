#date: 2023-01-25T16:46:26Z
#url: https://api.github.com/gists/f753fc531cba35d883c1c1a784a5e3eb
#owner: https://api.github.com/users/mBohunickaCharles

sentence_list = sentence.split(' ')
num_a_or_e = 0

for i in sentence_list:
    if ('a' in i) or ('e' in i):
        num_a_or_e += 1
        
print(num_a_or_e)        