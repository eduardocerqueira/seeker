#date: 2022-10-26T17:13:29Z
#url: https://api.github.com/gists/8fc14e985244c70f282a5024adbbc091
#owner: https://api.github.com/users/farracer004

##You have a list of your favourite marvel super heros.
#heros=['spider man','thor','hulk','iron man','captain america']
#Using this find out,


heros = ['spider man','thor','hulk','iron man','captain america']

#1. Length of the list
print(len(heros))
#2. Add 'black panther' at the end of this list
heros.append("black panther")
print(heros)
#3. You realize that you need to add 'black panther' after 'hulk',
   #so remove it from the list first and then add it after 'hulk'

heros.remove('black panther')
print(heros)
heros.insert(3, "black panther")
print(heros)

#4. Now you don't like thor and hulk because they get angry easily :)
# So you want to remove thor and hulk from list and replace them with doctor strange (because he is cool).
heros.remove("thor")
heros.remove("hulk")
heros.insert(1, "doctor strange")
print(heros)
   #Do that with one line of code.
#5. Sort the heros list in alphabetical order (Hint. Use dir() functions to list down all functions available in list)
heros.sort()
print(heros)
