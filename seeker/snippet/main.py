#date: 2022-04-26T16:59:22Z
#url: https://api.github.com/gists/d35751e1ba8f4009ba1592b0e78f2f0b
#owner: https://api.github.com/users/Naketsmall


sentences = input().split('.')
sentences = [sentence for sentence in sentences if sentence != '']
ans = ''
for sentence in sentences:
    words = sentence.split()
    words[0] = words[0].lower()
    words.reverse()
    words[0] = words[0].capitalize()
    for word in words:
        ans += word + ' '
    ans = ans[:-1] + '. '
print(ans)

