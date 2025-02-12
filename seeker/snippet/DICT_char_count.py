#date: 2025-02-12T16:54:11Z
#url: https://api.github.com/gists/a292f93a0b84f4ac1fa83837197d5b35
#owner: https://api.github.com/users/idenise

stri = "what can I do"
char_d={}
for c in stri:
      if c not in char_d:
             char_d[c] = 0
      char_d[c] = char_d[c] + 1    
print(char_d) 