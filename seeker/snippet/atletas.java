//date: 2024-05-13T16:50:05Z
//url: https://api.github.com/gists/a34ee4d5e43da5549697ef466b5d41ba
//owner: https://api.github.com/users/abfranca

import java.util.Locale;
import java.util.Scanner;

public class atletas {
    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        Scanner sc = new Scanner(System.in);

        System.out.print("Qual a quantidade de atletas? ");
        int quantidadeAtletas = sc.nextInt();
        sc.nextLine();

        char sexo;
        double alturaAtual, alturaMulheres = 0.0, maiorAltura = 0.0, pesoAtual, pesoTotal = 0.0;
        int quantidadeHomens = 0, quantidadeMulheres = 0;
        String nomeAtletaAtual, nomeAtletaMaisAlto = "";

        for (int i = 1; i <= quantidadeAtletas; i++) {
            System.out.printf("Digite os dados do atleta numero %d:", i);
            System.out.print("\nNome: ");
            nomeAtletaAtual = sc.nextLine();

            System.out.print("Sexo: ");
            do {
                sexo = sc.nextLine().charAt(0);

                if (sexo != 'F' && sexo != 'M') {
                    System.out.print("Valor invalido! Favor digitar F ou M: ");
                }
            } while (sexo != 'F' && sexo != 'M');

            System.out.print("Altura: ");
            do {
                alturaAtual = sc.nextDouble();

                if (alturaAtual <= 0.0) {
                    System.out.print("Valor invalido! Favor digitar um valor positivo: ");
                }
            } while (alturaAtual <= 0.0);

            if (sexo == 'F') {
                alturaMulheres += alturaAtual;
                quantidadeMulheres++;
            } else {
                quantidadeHomens++;
            }

            System.out.print("Peso: ");
            do {
                pesoAtual = sc.nextDouble();

                if (pesoAtual <= 0.0) {
                    System.out.print("Valor invalido! Favor digitar um valor positivo: ");
                }
            } while (pesoAtual <= 0.0);

            pesoTotal += pesoAtual;

            if (i == 1 || alturaAtual > maiorAltura) {
                maiorAltura = alturaAtual;
                nomeAtletaMaisAlto = nomeAtletaAtual;
            }
        }

        System.out.print("\nRELATÓRIO:");
        System.out.printf("\nPeso médio dos atletas: %.2f", pesoTotal / quantidadeAtletas);
        System.out.print("\nAtleta mais alto: " + nomeAtletaMaisAlto);
        System.out.printf("\nPorcentagem de homens: %.1f %%", (double) quantidadeHomens / (double) quantidadeAtletas * 100);

        if (quantidadeMulheres > 0) {
            System.out.printf("\nAltura média das mulheres: %.2f", alturaMulheres / quantidadeMulheres);
        } else {
            System.out.print("\nNão há mulheres cadastradas");
        }

        sc.close();
    }
}
