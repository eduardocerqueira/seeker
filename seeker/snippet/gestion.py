#date: 2025-11-20T16:58:38Z
#url: https://api.github.com/gists/623e674ba591ad5d2fb420b939639f9c
#owner: https://api.github.com/users/GaliumAlbum

def tableau_amortissement_echeance_constante(
    capital_initial: float,
    taux_annuel:     float,
    duree_annees:    int,
    periodicite:     int = 1
):
    """
    Calcule et affiche le tableau d'amortissement d'un prêt à échéance constante.

    :param capital_initial: Montant emprunté (ex: 100000)
    :param taux_annuel: Taux d'intérêt annuel en pourcentage (ex: 3 pour 3 %)
    :param duree_annees: Durée du prêt en années (ex: 5)
    :param periodicite: Nombre d'échéances par an (1=annuel, 12=mensuel, etc.)
    """
    # Taux et nombre total de périodes
    i = taux_annuel / 100 / periodicite      # taux par période (ex: 0.03/12)
    n = duree_annees * periodicite           # nombre total d'échéances

    # Calcul de l'échéance constante (formule d'annuité)
    echeance = capital_initial * i / (1 - (1 + i) ** -n)

    solde = capital_initial

    # En-tête du tableau
    print(f"{'Période':>7} | {'Capital début':>14} | {'Intérêt':>10} | {'Amortiss.':>11} | {'Échéance':>10} | {'Capital fin':>12}")
    print("-" * 78)

    for k in range(1, n + 1):
        interet = solde * i
        amortissement = echeance - interet
        solde = solde - amortissement

        print(
            f"{k:7d} | "
            f"{capital_initial if k == 1 else '':>14}",  # optionnel : tu peux afficher solde début ici
        )

    # Version plus simple (sans essayer d'être "trop joli") :
def tableau_amortissement_echeance_constante_simple(
    capital_initial: float,
    taux_annuel: float,
    duree_annees: int,
    periodicite: int = 1
):
    i = taux_annuel / 100 / periodicite
    n = duree_annees * periodicite
    echeance = capital_initial * i / (1 - (1 + i) ** -n)

    solde = capital_initial

    print(f"{'Période':>7} | {'Solde début':>12} | {'Intérêt':>10} | {'Amortiss.':>11} | {'Échéance':>10} | {'Solde fin':>10}")
    print("-" * 78)

    for k in range(1, n + 1):
        interet = solde * i
        amortissement = echeance - interet
        solde_fin = solde - amortissement

        print(
            f"{k:7d} | "
            f"{solde:12.2f} | "
            f"{interet:10.2f} | "
            f"{amortissement:11.2f} | "
            f"{echeance:10.2f} | "
            f"{solde_fin:10.2f}"
        )

        solde = solde_fin

    # Petite correction d'affichage si on est à quelques centimes de 0
    if abs(solde) < 0.01:
        print("\n(Remarque : le solde final est pratiquement 0, les écarts viennent des arrondis.)")


# Exemple d'utilisation pour ton cas :
if __name__ == "__main__":
    tableau_amortissement_echeance_constante_simple(
        capital_initial=100000,
        taux_annuel=3,
        duree_annees=5,
        periodicite=1  # 1 = annuités (échéances annuelles)
    )

