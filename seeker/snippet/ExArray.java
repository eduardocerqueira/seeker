//date: 2022-07-26T16:59:21Z
//url: https://api.github.com/gists/3dcf9ae2444b0381dd5d0307fd5e79fe
//owner: https://api.github.com/users/marianahenckel

import java.util.Scanner;

public class ExArray {
    public static void main(String[] args) {
        int[] lista = new int[5]; //Cria um array do tipo int com 5 itens

        Scanner leitura = new Scanner(System.in); //Ferramentas para entrada de dados

        //Leitura de dados escritas pelo usuario

        for (int i = 0; i < lista.length; i++) {
            //i é a variavel de controle, enquanto i for menor que a qtdade de itens da lista faça...
            //cada vez que roda o for i recebe ele mesmo + 1

            System.out.println("Insira um numero: "); // pedido ao usuario

            lista[i] = leitura.nextInt(); //Atribui a cado espaço da lista o que o usuario digitar
        }

        //Saida de dados com base na leitura
        for (int i = 0; i < lista.length; i++) {

            System.out.println(lista[i]); //print tudo o que esta no array
        }
    }
}
