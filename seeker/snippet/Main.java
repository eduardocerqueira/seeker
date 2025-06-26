//date: 2025-06-26T16:58:33Z
//url: https://api.github.com/gists/8ac5f52e141590a71807fef234503d33
//owner: https://api.github.com/users/LukasLS01

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class Main {
    public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        List<Aluno> alunos = new ArrayList<>();
        var p = System.out;
        alunos.add(new Aluno("Joao"));
        alunos.add(new Aluno("Ana"));
        alunos.add(new Aluno("Alberto"));
        alunos.add(new Aluno("Douglas"));
        alunos.add(new Aluno("Maria"));
        alunos.add(new Aluno("Bruna"));
        for (Aluno e : alunos){
            p.println("Nome: " + e.getNome() + " | Nota: " + e.getNota());
        }
        Collections.sort(alunos);
        p.println();
        for (Aluno e : alunos){
            p.println("Nome: " + e.getNome() + " | Nota: " + e.getNota());
        }
    }
}