//date: 2022-08-31T17:05:44Z
//url: https://api.github.com/gists/e95c50900b01b4f918d6dedd375e58af
//owner: https://api.github.com/users/jcarlosvale

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ExercicioClasseLambda {

    /* NAO MODIFICAR ESTE METODO*/
    public static List<String> getAlunos() {
        var strAlunos =
                "Artur Bezerra Delabio Ferraz\n" +
                "Cristiely Gomes Pires\n" +
                "Danielson Macedo Mesquita\n" +
                "Diego Gonzales D'almeida\n" +
                "Felipe Viana Santana\n" +
                "Hugo Moes Galvao Maciel\n" +
                "José Railson Barros Ribeiro\n" +
                "José Rodrigues Guimarães Filho\n" +
                "Lucas De Moraes Pinto Pereira\n" +
                "Luiz Filipe Moraes Saldanha Oliveira\n" +
                "Luiza Pinheiro Andrade Viana\n" +
                "Mariana Lemos Silva Lisboa\n" +
                "Mario Watanabe\n" +
                "Matheus Nantes Costa\n" +
                "Pedro Henrique Soares Da Silva\n" +
                "Renan Delgado Cavalcante Alves Mendes\n" +
                "Richard Da Silva Barroso\n" +
                "Victor Santos Pimentel Rodrigues De Araújo";
        var listAlunos =  new ArrayList<>(Arrays.asList(strAlunos.split("\n")));
        Collections.shuffle(listAlunos);
        return listAlunos;
    }

    public static void main(String[] args) {
        System.out.println("############# EXEMPLO LISTAGEM 1");
        System.out.println(getAlunos());
        System.out.println("############# EXEMPLO LISTAGEM 2");
        System.out.println(getAlunos());
        System.out.println("######## ALUNOS EM ORDEM ALFABETICA, POR LINHA: ");
        imprimirEmOrdemAlfabetica(getAlunos());
        System.out.println("######## ALUNAS DA TURMA, POR LINHA: ");
        imprimirAlunas(getAlunos());
    }

    private static void imprimirEmOrdemAlfabetica(List<String> alunos) {
        //TODO: implementar usando método .sort() para ordenar
        //TODO: implementar usando lambda expression e interface Comparator
        //TODO: usar método .forEach() para imprimir
    }

    private static void imprimirAlunas(List<String> alunos) {
        //TODO: implementar usando método .removeIf() para remover alunos
        //TODO: implementar o Predicate usando lambda expression
        //TODO: usar método .forEach() para imprimir
    }


}