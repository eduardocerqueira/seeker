#date: 2024-11-05T16:58:54Z
#url: https://api.github.com/gists/30f357288b547afed3e1b03125602d13
#owner: https://api.github.com/users/ThallyssonKlein

import bcrypt

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"h "**********"a "**********"s "**********"h "**********"e "**********"d "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"p "**********"l "**********"a "**********"i "**********"n "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    salt = bcrypt.gensalt()
    hashed_password = "**********"
    return hashed_password.decode('utf-8')

def main():
    num_users = int(input("Quantos usuários você deseja adicionar? "))
    inserts = []

    for _ in range(num_users):
        username = input("Digite o nome de usuário: ")
        password = input("Digite a senha: "**********"
        hashed_password = "**********"
        inserts.append(f"INSERT INTO users (username, password) VALUES ('{username}', '{hashed_password}');")

    print("\nComandos SQL gerados:")
    for insert in inserts:
        print(insert)

if __name__ == "__main__":
    main()