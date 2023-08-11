//date: 2023-08-11T16:52:22Z
//url: https://api.github.com/gists/21bc01b61453416ce1950ab72ca70651
//owner: https://api.github.com/users/khalillakhdhar

import java.util.Scanner;

public class SuiteMathematique {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Entrez le nombre de termes à générer : ");
        int n = scanner.nextInt();
        scanner.close();

        int terme = 1;

        for (int i = 1; i <= n; i++) {
            System.out.println("Terme " + i + " : " + terme);

            if (i % 2 == 1) {
                terme = terme + 2;
            } else {
                terme = terme * 2;
            }
        }
    }
}
