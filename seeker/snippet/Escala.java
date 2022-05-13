//date: 2022-05-13T17:00:56Z
//url: https://api.github.com/gists/a64998e830ae5d91ef62556d3c05d393
//owner: https://api.github.com/users/gomesgr

import java.util.ArrayList;
import java.util.List;

public class Escala {
    private final List<NotaMusical> notas = List.of(NotaMusical.values());
    // Escala maior (natural) eh composta pela sequencia de
    // 1ra, Tom, Tom, STom, Tom, Tom, Tom, STom
    // Do          Re, Mi, Fa, Sol, La, Si, Do <- Oitava

    // Escala menor (natural) eh composta pela sequencia de
    // 1ra, Tom, STom, Tom, Tom, STom, Tom, Tom


    // Metodo que da uma nota e constroir a partir dela
    // uma escala maior, seguindo a sequencia
    public List<NotaMusical> maiorNatural(NotaMusical notaMusical) {
        // c,cs,d,ds,e,f,fs,g,gs,a,as,b
        // 0,1 ,2,3 ,4,5,6 ,7,8 ,9,10,11
        int index = notas.indexOf(notaMusical);
        int meios = 0;
        List<NotaMusical> escalaMaior = new ArrayList<>();
        // Cada passo apesar de aumentar de 1 em 1 vale meio na escala
        /*
         * meio 0 C
         * meio 1 Cs
         * meio 2 D
         * meio 3 Ds
         * meio 4 E
         * meio 5 F
         * meio 6 Fs
         * meio 7 G
         * meio 8 Gs
         * meio 9 A
         * meio 10 As
         * meio 11 B
         */
        while(escalaMaior.size() < 8) {
            if (meios == 12) {
                meios = 0;
            }
            if (index >= notas.size())
                index = 0;
            switch (meios) {
                case 0, 2, 4, 5, 7, 9, 11 -> escalaMaior.add(notas.get(index));
            }
            index++;
            meios++;
        }
    return escalaMaior;
    }

    // Metodo para criar uma escala menor seguindo a sequencia
    public List<NotaMusical> menorNatural(NotaMusical notaMusical) {
        List<NotaMusical> escalaMenorNatural = new ArrayList<>();
        int meios = 0;
        int index = notas.indexOf(notaMusical);
        while(escalaMenorNatural.size() < 8) {
            if (meios == 12)
                meios = 0;
            if (index >= notas.size())
                index = 0;

            switch (meios) {
                case 0, 2, 3, 5, 7, 8, 10 -> escalaMenorNatural.add(notas.get(index));
            }
            meios++;
            index++;
        }
        return escalaMenorNatural;
    }

    public List<NotaMusical> menorHarmonica(NotaMusical notaMusical) {
        List<NotaMusical> escalaMenorHarmonica = new ArrayList<>();
        int meios = 0;
        int index = notas.indexOf(notaMusical);
        while(escalaMenorHarmonica.size() < 8) {
            if (meios == 12)
                meios = 0;
            if (index >= notas.size())
                index = 0;

            switch (meios) {
                case 0, 2, 3, 5, 7, 8, 11 -> escalaMenorHarmonica.add(notas.get(index));
            }
            meios++;
            index++;
        }
        return escalaMenorHarmonica;
    }
}

// Enum das Notas Musicais
public enum NotaMusical {
    C("Dó"),
    Cs("Dó sustenido"),
    D("Ré"),
    Ds("Ré sustenido"),
    E("Mi"),
    F("Fá"),
    Fs("Fá sustenido"),
    G("Sol"),
    Gs("Sol sustenido"),
    A("Lá"),
    As("Lá sustenido"),
    B("Si");

    private final String nome;

    NotaMusical(String nome) {
        this.nome = nome;
    }

    public String nome() {
        return nome;
    }
}