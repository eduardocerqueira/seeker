#date: 2022-02-21T16:56:09Z
#url: https://api.github.com/gists/9040795167892decfa6b05e80099b8c0
#owner: https://api.github.com/users/machinemaniac

stopwords = ['to', 'a', 'for', 'by', 'an', 'am', 'the', 'so', 'it', 'and', 'The']
sent = "The water earth and air are vital"
brk = sent.split(" ")
for ch in brk:
 if ch in stopwords:
  stopwords.append(ch)
  brk.remove(ch)
caps = " "
for ch in brk:
 caps = caps + " " + ch[0:2].upper() + "."
acro = caps[0:-1]
print(acro)