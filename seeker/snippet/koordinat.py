#date: 2022-12-16T16:39:39Z
#url: https://api.github.com/gists/4ce37984956e15fd1cb226c85dc47f8f
#owner: https://api.github.com/users/edib

x, y = input("x ve y sayılarını girin: ").split()

x = int(x)
y = int(y)

if (x > 0 and y >0 ):
    print("{} ve {} sayıları I. alandadır.".format(x,y)) 
elif (x < 0 and y > 0):
    print("{} ve {} sayıları II. alandadır.".format(x,y))
elif (x < 0 and y < 0):
    print("{} ve {} sayıları III. alandadır.".format(x,y))    
elif (x > 0 and y < 0):
    print("{} ve {} sayıları IV. alandadır.".format(x,y))        
else:
    print("{} ve {} sayıları orijindedir.".format(x,y))        