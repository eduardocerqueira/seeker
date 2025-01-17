#date: 2025-01-17T17:03:52Z
#url: https://api.github.com/gists/5399fcf9fccea899ae599a10b425cba4
#owner: https://api.github.com/users/MazenEmadAbdelsattar

#1st dictionary

Info = {
    "mazen" : "Python",
    "saad" : "Java",
    "ali" : "Css",
    "samer" : "HTML",
    "yasser" : "C++"
}

for student in Info :
    print(f"{student.strip().upper()} from our team, He specialised in [ {Info[student].capitalize()} ]")


#2nd dictionary

Univesrsity = {
    "jasmine" :["Cs","IS","Math_1"],
    "mazen" :["Physics","IS","Cs"],
    "yasser" :["Stat_1","English","CS"],
    "khaled" :["Is","Cs","English"],
    "joe" :["Cs","Is","Stat_1"],
    "sara" :["English","IS","Cs"],
}

for student in Univesrsity :
    print (f"{student.upper().strip()} is student at Helwan university and subject of this year => {Univesrsity[student]}")