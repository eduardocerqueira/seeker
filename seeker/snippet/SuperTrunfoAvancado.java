//date: 2025-09-09T16:49:28Z
//url: https://api.github.com/gists/abffd98e4e6a832a1ed145e0ddc9fc46
//owner: https://api.github.com/users/FelipeAvelinoRO

import java.util.Locale;
import java.util.Scanner;

public class SuperTrunfoAvancado {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        Scanner sc = new Scanner(System.in);

        // Variáveis de cada carta
        char[] estados = new char[2];
        String[] codigos = new String[2];
        String[] cidades = new String[2];
        long[] populacoes = new long[2];
        float[] areas = new float[2];
        float[] pibs = new float[2]; // PIB em bilhões de reais
        int[] pontosTuristicos = new int[2];
        float[] densidades = new float[2];
        float[] pibPerCapita = new float[2];
        float[] superPoder = new float[2];

        // Cadastro das 2 cartas
        for (int i = 0; i < 2; i++) {
            System.out.println("\n--- Cadastro da Carta " + (i + 1) + " ---");

            System.out.print("Digite o Estado (A-H): ");
            estados[i] = sc.next().charAt(0);

            System.out.print("Digite o Código da Carta (ex: A01, B03): ");
            codigos[i] = sc.next();

            System.out.print("Digite o Nome da Cidade: ");
            sc.nextLine();
            cidades[i] = sc.nextLine();

            System.out.print("Digite a População da Cidade: ");
            long pop = sc.nextLong();
            if (pop < 0) {
                System.out.println("Valor inválido! População deve ser positiva.");
                return;
            }
            populacoes[i] = pop;

            System.out.print("Digite a Área da Cidade (em km²): ");
            areas[i] = sc.nextFloat();

            System.out.print("Digite o PIB da Cidade (em bilhões de reais): ");
            pibs[i] = sc.nextFloat();

            System.out.print("Digite o Número de Pontos Turísticos: ");
            pontosTuristicos[i] = sc.nextInt();

            // ---- Cálculos ----
            densidades[i] = populacoes[i] / areas[i];
            pibPerCapita[i] = (pibs[i] * 1_000_000_000f) / populacoes[i];

            // Super Poder = soma dos atributos numéricos
            // Inclui: população, área, PIB, pontos turísticos, PIB per capita, 1/densidade
            superPoder[i] = populacoes[i]
                    + areas[i]
                    + pibs[i]
                    + pontosTuristicos[i]
                    + pibPerCapita[i]
                    + (1 / densidades[i]);
        }

        // ---- Comparação ----
        System.out.println("\n===== Comparação de Cartas =====");

        // População
        System.out.println("População: Carta 1 venceu (" + (populacoes[0] > populacoes[1] ? 1 : 0) + ")");

        // Área
        System.out.println("Área: Carta 1 venceu (" + (areas[0] > areas[1] ? 1 : 0) + ")");

        // PIB
        System.out.println("PIB: Carta 1 venceu (" + (pibs[0] > pibs[1] ? 1 : 0) + ")");

        // Pontos Turísticos
        System.out.println("Pontos Turísticos: Carta 1 venceu (" + (pontosTuristicos[0] > pontosTuristicos[1] ? 1 : 0) + ")");

        // Densidade Populacional (menor vence)
        System.out.println("Densidade Populacional: Carta 1 venceu (" + (densidades[0] < densidades[1] ? 1 : 0) + ")");

        // PIB per Capita
        System.out.println("PIB per Capita: Carta 1 venceu (" + (pibPerCapita[0] > pibPerCapita[1] ? 1 : 0) + ")");

        // Super Poder
        System.out.println("Super Poder: Carta 1 venceu (" + (superPoder[0] > superPoder[1] ? 1 : 0) + ")");

        sc.close();
    }
}
