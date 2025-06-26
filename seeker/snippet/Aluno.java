//date: 2025-06-26T16:58:33Z
//url: https://api.github.com/gists/8ac5f52e141590a71807fef234503d33
//owner: https://api.github.com/users/LukasLS01

import java.util.*;

    public class Aluno implements Comparable<Aluno>{
        private String nome;
        private int nota;

    public Aluno (String nome){
            this.nome = nome;

            Random rdm = new Random();
            this.nota = rdm.nextInt(0, 10);
        }

    public String getNome() {
            return nome;
        }

    public int getNota() {
            return nota;
        }

    @Override
    public int compareTo(Aluno o) {
        int comp = Integer.compare(this.nota, o.nota);
            if (comp != 0)
                return comp;
        return this.nome.compareTo(o.nome);
    }
}


