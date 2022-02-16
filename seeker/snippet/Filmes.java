//date: 2022-02-16T17:06:18Z
//url: https://api.github.com/gists/7e79a7b8b3ae83a8c3f9723cccefae06
//owner: https://api.github.com/users/ramonsl

 import java.util.Arrays;

public class Filmes {
    public static final double maximo = 3.00;

    public static void main(String[] args) {
      //  double duracoes[] = {1.90, 1.04, 1.25, 2.5, 1.75};
        //double duracoes[] = {1, 1, 1, 1, 1};
        //double duracoes[] = {1, 2, 1, 2, 1};
         double duracoes[] = {2.5, 2.5, 2.5, 2.5, 1.75};
        int dias = acharMinimoDeDias(duracoes);
        System.out.println("Minimo de dias :" + dias);

    }

    private static int acharMinimoDeDias(double filmes[]) {
        int dias = 1;
        double tempo = 0;
        ordenar(filmes);
        for (int i = 0; i < filmes.length; i++) {
            if (tempo + filmes[i] <= maximo) {
                tempo = tempo + filmes[i];
            } else {
                tempo = 0;
                dias++;
                i--;
            }
        }
        return dias;
    }

    private static void ordenar(double filmes[]) {
        Arrays.sort(filmes);
    }

}
