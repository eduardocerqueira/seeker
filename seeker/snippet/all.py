#date: 2021-10-11T17:04:47Z
#url: https://api.github.com/gists/a0d7ba0e81fb9d0d706fc7723d1c5baa
#owner: https://api.github.com/users/kirilsSurovovs

name = input("Ievadiet savu vārdu:\n")
new_name = name[::-1].capitalize()
print(f"{new_name}, pamatīgs juceklis vai ne {name[0]}?")

text = input("Ievadiet kādu tekstu:\n")
allways_show_symbols = ' 123456890'
symbol = 'start'
while symbol != 'exit':
    new_text = ''
    for i in range(len(text)): # šajā ciklā izveido new_text, kas saturēs doto simbolu un zvaigznītes
        if text[i] == symbol or text[i] in allways_show_symbols:
            new_text += text[i]
        elif text[i] == symbol.upper(): # pārbauda arī lielo burtu, ja ievadīts mazais...
            new_text += symbol.upper()
        elif text[i] == symbol.lower(): # ...un mazo, ja ievadīts lielais
            new_text += symbol.lower()
        else:
            new_text += '*'
    print(new_text)
    symbol = input('Ievadiet burtu, ko vajag parādīt; vai ievadiet "exit", lai beigtu spēli: ')

text = input("Ievadiet kādu tekstu:\n")
if "nav" in text and "slikts" in text:
    # pieņēmu, ka posms "nav...slikts" ir tikai viens
    nav_index = text.rfind("nav") # šeit pieņēmu, ka meklējam starp pēdējo vārdu "nav", jo tādi var būt vairāki
    slikts_index = text[nav_index:].find("slikts") # un līdz pirmajam vārdam "slikts", kas parādās pēc tam
    if slikts_index>0: # pārbaudu, vai vispār ir atrasts
        print(text[:nav_index] + "ir labs" + text[nav_index+slikts_index+6:])
    else:
        # jāpārbauda arī gadījums, kad "slikts" parādās pirms pēdējā "nav", piemēram "Laiks nav slikts, bet laika nav"
        slikts_index = text.find("slikts")  # šeit atrodu pirmo "slikts"
        nav_index = text[:slikts_index].rfind("nav") # atrodu "nav" pirms tam
        if nav_index > -1: # pārbaudu, vai ir atrasts šajā posmā
            print(text[:nav_index] + "ir labs" + text[nav_index + slikts_index:])
        else:
            print(text)
else:
    print(text)