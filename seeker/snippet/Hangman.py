#date: 2022-03-10T17:08:18Z
#url: https://api.github.com/gists/2a185de0d381f39c3c6c25c1a5cf2223
#owner: https://api.github.com/users/rohit988

word = ['about', 'actually', 'after', 'again', 'against', 'always', 'among', 'animal', 'another', 'answer', 'appear', 'beauty', 'because', 'become', 'before', 'began', 'begin', 'behind', 'better', 'better', 'between', 'black', 'bottom', 'bring', 'brought', 'build', 'built', 'carefully', 'carry', 'centre', 'certain', 'change', 'check', 'child', 'children', 'class', 'clear', 'close', 'colour', 'common', 'community', 'complete', 'contain', 'could', 'country', 'course', 'create', 'cried', 'cross', 'decide', 'decided', 'develop', 'didn’t', 'different', 'don’t', 'dream', 'drive', 'during', 'early', 'earth', 'effort', 'enough', 'every', 'example', 'experience', 'explain', 'false', 'family', 'father', 'field', 'first', 'follow', 'found', 'friend', 'front', 'government', 'great', 'green', 'ground', 'group', 'happen', 'happened', 'heavy', 'horse', 'house', 'hundred', 'I, J, K', 'important', 'include', 'island', 'known', 'language', 'large', 'later', 'laugh', 'learn', 'leave', 'letter', 'light', 'listen', 'little', 'machine', 'measure', 'might', 'million', 'minute', 'money', 'month', 'morning', 'mother', 'mountain']
import random
random_number = random.choice(word)
random_number_list = [m for m in random_number]
len_word = len(random_number)
underline = ['-' for _ in range(int(len_word))]
list_user_words = []
underline_string = ''

def word_checker():
    global underline_string
    user_input = input('enter a word')
    list_user_words.append(user_input)
    i = 0
    for x in random_number_list:
        if user_input ==x:
            underline[i] = x
            underline_string = ''.join(underline)
        else:
            pass
        i += 1
    return underline_string
x = ''

while x != random_number:
    x = word_checker()
    print(f'you have used these words {list_user_words}')
    print(x)
print(f"Hooray ! you have done it")




    

