//date: 2025-08-19T17:07:40Z
//url: https://api.github.com/gists/a89ce61ba807730f9932f3aef0d95dcd
//owner: https://api.github.com/users/caique-moura

import java.util.Scanner;

public class Main {
    public static void main(String args[] ){
        Scanner sc = new Scanner (System.in);

        System.out.println("BEM VINDO À CALCULADORA");

        System.out.print("Digite o primeiro número: ");
        double n1 = sc.nextDouble();

        System.out.print("Digite o segundo número: ");
        double n2 = sc.nextDouble();

        double somaN = n1 + n2;
        double divisaoN = n1 / n2;
        double subtracaoN = n1 - n2;
        double multiN = n1 * n2;

        System.out.println("Soma: " + somaN);
        System.out.println("Subtração: " + subtracaoN);
        System.out.println("Divisão: " + divisaoN);
        System.out.println("Multiplicação: " + multiN);

        sc.close();

    }
}