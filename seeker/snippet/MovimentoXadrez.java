//date: 2025-09-09T17:00:15Z
//url: https://api.github.com/gists/63796729719779a2a62d64864af7150e
//owner: https://api.github.com/users/FelipeAvelinoRO

public class MovimentoXadrez {

    public static void main(String[] args) {

        // ==============================
        // Movimento da TORRE (usando FOR)
        // ==============================

        int casasTorre = 5; // número de casas que a torre irá se mover
        System.out.println("Movimento da Torre:");
        for (int i = 1; i <= casasTorre; i++) {
            System.out.printf("Direita\n");
        }

        // ==============================
        // Movimento do BISPO (usando WHILE)
        // ==============================

        int casasBispo = 5; // número de casas que o bispo irá se mover
        int contadorBispo = 1;
        System.out.println("\nMovimento do Bispo:");
        while (contadorBispo <= casasBispo) {
            System.out.printf("Cima Direita\n");
            contadorBispo++;
        }

        // ==============================
        // Movimento da RAINHA (usando DO-WHILE)
        // ==============================

        int casasRainha = 8; // número de casas que a rainha irá se mover
        int contadorRainha = 1;
        System.out.println("\nMovimento da Rainha:");
        do {
            System.out.printf("Esquerda\n");
            contadorRainha++;
        } while (contadorRainha <= casasRainha);

        // ==============================
        // Movimento do CAVALO (usando FOR + WHILE aninhados)
        // ==============================

        int casasBaixo = 2;     // número de casas para baixo
        int casasEsquerda = 1;  // número de casas para esquerda

        System.out.println("\nMovimento do Cavalo:");

        // Primeiro movimento: duas casas para baixo (usando for)

        for (int i = 1; i <= casasBaixo; i++) {
            System.out.printf("Baixo\n");
        }

        // Depois, uma casa para esquerda (usando while dentro do bloco)

        int contadorEsquerda = 0;
        while (contadorEsquerda < casasEsquerda) {
            System.out.printf("Esquerda\n");
            contadorEsquerda++;
            
        }
    }
}
