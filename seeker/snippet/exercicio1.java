//date: 2022-11-18T16:58:15Z
//url: https://api.github.com/gists/2a1f7e9ffb434876888f3517ba60521a
//owner: https://api.github.com/users/anabeatrizdmt

package aula4_20221116;

import java.util.Scanner;

public class exercicio1 {

    public static void main(String[] args) {

        String resultado[][] = {
                {String.valueOf(1),"Sem direito a votar"},
                {String.valueOf(10),"Sem direito a votar"},
                {String.valueOf(15),"Sem direito a votar"},
                {String.valueOf(16),"Voto facultativo"},
                {String.valueOf(17),"Voto facultativo"},
                {String.valueOf(70),"Voto facultativo"},
                {String.valueOf(18),"Voto obrigatório"},
                {String.valueOf(44),"Voto obrigatório"},
                {String.valueOf(69),"Voto obrigatório"}
        };

        for (int i = 0; i < resultado.length; i++) {
            if (regraVoto(Integer.parseInt(resultado[i][0])).equals(resultado[i][1])) {
                System.out.println("Test "+ (i+1) + ": Age " + resultado[i][0] + " should return " + resultado[i][1] + ": PASSED");
            } else {
                System.err.println("Test "+ (i+1) + ": Age " + resultado[i][0] + " should return " + resultado[i][1] + ": FAILED. It returned: " + regraVoto(Integer.parseInt(resultado[i][0])));
            }
        }
    }

    public static String regraVoto(int idade) {
        if (idade < 16) {
            return "Sem direito a votar";
        } else if ((idade >= 16 && idade < 18) || idade >= 70) {
            return "Voto facultativo";
        } else {
            return "Voto obrigatório";
        }
    }

    public static int getIdade() {
        Scanner scan = new Scanner(System.in);

        Boolean idadeInvalida = true;

        int idade = 0;
        while (idadeInvalida) {
            try {
                System.out.println("Insira sua idade: ");
                idade = Integer.parseInt(scan.nextLine());
                idadeInvalida = false;
            } catch (NumberFormatException e) {
                System.out.println("Insira apenas um número inteiro, por favor.");
            } catch (Exception e) {
                System.out.println("Erro " + e);
            }
        }
        scan.close();
        return idade;
    }
}
