//date: 2025-09-08T17:01:20Z
//url: https://api.github.com/gists/9c00e425e04dc1e9d0cdf254c03e1ec4
//owner: https://api.github.com/users/FelipeAvelinoRO

import java.util.Locale;
import java.util.Scanner;

public class SuperTrunfo {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        Scanner sc = new Scanner(System.in);

        // Vamos armazenar 2 cartas
        for (int i = 1; i <= 2; i++) {
            System.out.println("\n--- Cadastro da Carta " + i + " ---");

            // Estado
            System.out.print("Digite o Estado (A-H): ");
            char estado = sc.next().charAt(0);

            // Código da Carta
            System.out.print("Digite o Código da Carta (ex: A01, B03): ");
            String codigoCarta = sc.next();

            // Nome da Cidade
            System.out.print("Digite o Nome da Cidade: ");
            sc.nextLine(); // consumir quebra de linha pendente
            String nomeCidade = sc.nextLine();

            // População
            System.out.print("Digite a População da Cidade: ");
            int populacao = sc.nextInt();

            // Área
            System.out.print("Digite a Área da Cidade (em km²): ");
            float area = sc.nextFloat();

            // PIB
            System.out.print("Digite o PIB da Cidade(em bilhões de reais): ");
            float pib = sc.nextFloat();

            // Pontos turísticos
            System.out.print("Digite o Número de Pontos Turísticos: ");
            int pontosTuristicos = sc.nextInt();

            // Exibir informações da carta
            System.out.println("\n===== CARTA " + i + " =====");
            System.out.println("Estado: " + estado);
            System.out.println("Código da Carta: " + codigoCarta);
            System.out.println("Nome da Cidade: " + nomeCidade);
            System.out.println("População: " + populacao);
            System.out.println("Área (km²): " + area);
            System.out.println("PIB: " + pib + " bilhões de reais");
            System.out.println("Número de Pontos Turísticos: " + pontosTuristicos);
            System.out.println("========================\n");
        }

        sc.close();
    }
}
