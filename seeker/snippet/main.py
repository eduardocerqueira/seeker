#date: 2022-03-25T17:01:20Z
#url: https://api.github.com/gists/43c3d0c42b8e1d9fae6f84557aa3000e
#owner: https://api.github.com/users/Emmremme23

import webbrowser

print("--Arama Aracı-- (Ne istersin? derse kapat yazınız işiniz bittiyse)")


def search_it(search):
    url = "https://google.com/search?q=" + search
    webbrowser.get().open(url)
    print(search + " Yükleniyor.")


while True:
    search = input("Ne İstersin:")
    if search == "kapat":
        print("İyi günler.")
        break
    else:
        search_it(search)



