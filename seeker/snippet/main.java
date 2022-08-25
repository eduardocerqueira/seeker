//date: 2022-08-25T15:21:22Z
//url: https://api.github.com/gists/dc1c72c8e2f36f88eb8c50cd3760b993
//owner: https://api.github.com/users/VladimirCGF

import java.util.Locale;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        Locale.setDefault(Locale.US);
        Scanner sc = new Scanner(System.in);

        double pi = 3.14159;
        double raio;
        double valorFinal;

        System.out.println("Digite o valor do raio: ");
        raio = sc.nextDouble();
        System.out.println("Fórmula da área: area = π . raio2");

        raio = raio * raio;
        valorFinal = pi * raio;

        System.out.println("Valor Final: " + valorFinal);


        
        sc.close();
    }
}
