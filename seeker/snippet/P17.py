#date: 2024-05-30T16:40:47Z
#url: https://api.github.com/gists/283f45df303186c667c79931e90ce773
#owner: https://api.github.com/users/VedaCPatel

libraryRecord = [ ["105MS" , "Marcus" , "Smith" , 25 ], ["103AZ" , "Anthony" , "Zarrent" , 5 ], ["108MW" , "Matt" , "White" , 12 ], ["112DB" , "Denise" , "Bilton" , 58 ], ["124MK" , "Malcolm" , "Kelly" , 26 ], ["116UK" , "Uzere" , "Kevill" , 29 ], ["127AL" , "Abduraheim" , "Leahy" , 94 ], ["124LS" , "Laura" , "Sampras" , 50 ], ["121AP" , "Azra" , "Potter" , 61 ], ["115AC" , "Anthony" , "Calik" , 10 ], ["117PI" , "Pablo" , "Iilyas" , 49 ], ["113MM" , "Mark" , "Montgomerie" , 68 ], ["130FH" , "Felicity" , "Heath" , 11 ], ["132JA" , "Jill" , "Alexander" , 61 ], ["123SG" , "Sara" , "Grimstow" , 9 ], ["134KD" , "Kevin" , "Dawson" , 74 ], ["122AB" , "Andrew" , "Bertwistle" , 42 ], ["125JF" , "Jaide" , "Feehily" , 55 ], ["128JS" , "Justin" , "Slater" , 68 ], ["126CG" , "Colleen" , "Grohl" , 39 ] ]
total=0
n=[]
for i in libraryRecord:
    total+=i[3]
    N=n.append(i[3])
average=total/2
print("The total amount of books read is:",total)
print("Average amount of books read:",average)
print("The IDs of students who have read less than 10 books are:")
b=0
for j in libraryRecord:
    if j[3]<10:
        print(j[0])
        b+=1
if b==0:
    print("There are no such students!")
M=n.sort()
l=[]
for k in n[-1::-1]:
    L=l.append(k)
for i in libraryRecord:
    if i[3]==l[0]:
        print("The gold member is:",str(i[1]),str(i[2]),"with",str(i[3]),"books!")
    elif i[3]==l[1]:
        print("The silver member is:",i[1],i[2],"with",i[3],"books!")
    elif i[3]==l[2]:
        print("The bronze member is:",i[1],i[2],"with",i[3],"books!")







