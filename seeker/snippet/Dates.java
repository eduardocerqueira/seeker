//date: 2023-08-11T16:52:22Z
//url: https://api.github.com/gists/21bc01b61453416ce1950ab72ca70651
//owner: https://api.github.com/users/khalillakhdhar

import java.util.Scanner;

public class DateValidator {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Entrez le jour : ");
        int jour = scanner.nextInt();

        System.out.print("Entrez le mois : ");
        int mois = scanner.nextInt();

        System.out.print("Entrez l'année : ");
        int annee = scanner.nextInt();

        scanner.close();

        boolean dateValide = verifierDate(jour, mois, annee);

        if (dateValide) {
            System.out.println("La date est valide !");
        } else {
            System.out.println("La date n'est pas valide.");
        }
    }

    public static boolean verifierDate(int jour, int mois, int annee) {
        if (mois < 1 || mois > 12 || jour < 1 || annee < 1) {
            return false; // Mois, jour ou année invalides
        }

        int joursDansMois;

        switch (mois) {
            case 2: // Février
                if (annee % 4 == 0 && (annee % 100 != 0 || annee % 400 == 0)) {
                    joursDansMois = 29; // Année bissextile
                } else {
                    joursDansMois = 28;
                }
                break;
            case 4:
            case 6:
            case 9:
            case 11:
                joursDansMois = 30; // Mois avec 30 jours
                break;
            default:
                joursDansMois = 31; // Mois avec 31 jours
        }

        return jour <= joursDansMois;
    }
}
