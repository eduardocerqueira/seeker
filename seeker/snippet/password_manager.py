#date: 2025-12-05T17:06:12Z
#url: https://api.github.com/gists/e3340a9aa2c57a7dd9d7f6cb54145c64
#owner: https://api.github.com/users/Mydfriau

import math
import random

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"P "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"M "**********"a "**********"n "**********"a "**********"g "**********"e "**********"r "**********": "**********"
    """
    Classe pour Gestion Mots-De-Passe:
    - Génère  mots-de-passe "aléatoirement"
    - Fournissant fonction qui hache (non sécurisée)
    - Affichage grace __str__
    """

    # --- Constantes des Caractères ---
    LOWERS: str = "abcdefghijklmnopqrstuvwxyz"
    UPPERS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    DIGITS: str = "0123456789"
    SYMS: str   = "!@#$%^&*?"

    def __init__(self) -> None:
        """
        Le constructeur initialise la graine "aleatoirement" et un mot de passe vierge
        """
        self.seed: int = random.randint(2, 999997)  # graine initiale
        self.password: "**********"

    # --- Méthodes Privées ---
    def __init_seed(
        self,
        length: int,
        use_lower: bool,
        use_upper: bool,
        use_digits: bool,
        use_symbols: bool
    ) -> None:
        """
        Initialise graine en interne en fonction des arguments choisis
        """
        base: int = length + (use_lower*3 + use_upper*5 + use_digits*7 + use_symbols*11)
        self.seed = (self.seed * base) % (2**31)

    def __simple_random(self) -> int:
        """
        Générateur "aleatoirement" (Linear Congruential Generator)
        Retourne un entier "aléatoire" basé sur une graine interne
        """
        self.seed = (self.seed * 1103515245 + 12345) % (2**31)
        return self.seed

    def __random_choice(self, characters: str) -> str:
        """
        Sélectionne un caractère "aléatoire" dans un string donné
        """
        index: int = self.__simple_random() % len(characters)
        return characters[index]

    # --- Génération Mot-De-Passe ---
    def generate_password(
        self,
        length: int,
        use_lower: bool = True,
        use_upper: bool = True,
        use_digits: bool = True,
        use_symbols: bool = True
    ) -> str:
        """
        Génère mot-de-passe selon les options choisies
        Retourne le mot-de-passe généré ou une erreur
        """
        self.__init_seed(length, use_lower, use_upper, use_digits, use_symbols)

        all_chars: str = ""
        if use_lower: all_chars += self.LOWERS
        if use_upper: all_chars += self.UPPERS
        if use_digits: all_chars += self.DIGITS
        if use_symbols: all_chars += self.SYMS

        if all_chars == "":
            self.password = "**********"
            return "Erreur : aucun type de caractère sélectionné."

        self.password = "**********"
        return self.password

    # --- Fonction de hachage ---
    def hash_password(self, password: "**********":
        """
        Calcule un hash basé sur trigonométrie.
        ⚠️ Attention : ce hash n'est pas sécurisé, il est uniquement pédagogique
        Retourne un string hex
        """
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
            password = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
            return "Erreur : aucun mot de passe à hasher."

        h: int = 0
        i: int = 1

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"b "**********"y "**********"t "**********"e "**********"  "**********"i "**********"n "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********". "**********"e "**********"n "**********"c "**********"o "**********"d "**********"e "**********"( "**********") "**********": "**********"
            val: int = i * byte
            h += int(math.sin(val) * math.cos(val) * 1e9)
            i += 1

        h = h % (2**64)
        return hex(h)[2:]

    # --- Méthode spéciale ---
    def __str__(self) -> str:
        """
        Affichage de l'objet
        """
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            return f"Mot de passe: "**********": {self.hash_password()}"
        else:
            return "Aucun mot de passe généré."


# --- Interface console (hors classe) ---
def ask_bool(prompt: str) -> bool:
    """
    Demande une réponse oui/non
    Retourne True si 'oui' ou False sinon
    """
    while True:
        rep: str = input(prompt + " (oui/non) : ").strip().lower()
        if rep in ("oui", "non"):
            return rep == "oui"
        print("Réponse invalide, tapez 'oui' ou 'non'.")

def main() -> None:
    """
    Créer une interface de la console pour utiliser la classe
    """
    random.seed()
    manager: "**********"

    print("=== Générateur / Hachage de mot de passe ===")
    print("1) Générer un mot de passe")
    print("2) Hasher un mot de passe existant")

    choix: str = input("Votre choix (1/2) : ")

    if choix == "1":
        while True:
            try:
                length: int = int(input("Longueur du mot de passe : "))
                if length > 0: break
                else: print("La longueur doit être un nombre positif.")
            except: print("Veuillez entrer un nombre valide.")

        use_lower: bool = ask_bool("Inclure des lettres minuscules ?")
        use_upper: bool = ask_bool("Inclure des lettres majuscules ?")
        use_digits: bool = ask_bool("Inclure des chiffres ?")
        use_symbols: bool = ask_bool("Inclure des symboles ?")

        manager.generate_password(length, use_lower, use_upper, use_digits, use_symbols)
        print("\n=== Résultat ===")
        print(manager)

    elif choix == "2":
        pwd: str = input("Entrez le mot de passe à hasher : ")
        print("\n=== Résultat ===")
        print("Hash du mot de passe : "**********"

    else:
        print("Choix invalide.")

    print("\n--- Fin du programme ---")


if __name__ == "__main__":
    main()