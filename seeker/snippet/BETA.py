#date: 2023-10-12T17:01:07Z
#url: https://api.github.com/gists/8a8cd08b1d9a69cf3f6fcf870b07163b
#owner: https://api.github.com/users/taiga-junior


def main():
    # recolter la premire note
    note1 = int(input("entre la premier note"))
    # recoltre la deuxiéme note
    note2 = int(input("etre la second note"))
    # recolter la troisieme note
    note3 = int(input("entre la derniere note"))
    # calculer la moyenne
    result = (note1 + note2 + note3) / 3
    # afficher le resultat
    print("la moyenne est de" + str(result))


if __name__ == '__main__':
    main()
