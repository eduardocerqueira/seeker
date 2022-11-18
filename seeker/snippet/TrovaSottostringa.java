//date: 2022-11-18T17:07:46Z
//url: https://api.github.com/gists/2f5c43d75304e852198de570bb93cdce
//owner: https://api.github.com/users/Mattia1992Dev

import java.util.Scanner;

public class TrovaSottostringa {

    Scanner scanner = new Scanner(System.in);
    private String frase="";

    public void trovaParola(){
        System.out.println("Inserisci frase ");

        frase= scanner.nextLine().toUpperCase();

        if(frase.contains("PARTITO"))
        {
            System.out.println("La frase digitata era: " + frase);

            frase= frase.replace("PARTITO", "TORNATO");

            System.out.println("la frase nuova è: " + frase);
        }

    }



}
