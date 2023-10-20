//date: 2023-10-20T17:01:57Z
//url: https://api.github.com/gists/485e40f11fc8d30b12ac5a22414fa8c0
//owner: https://api.github.com/users/canaldojavao

package br.com.escola.gestaoescolar;

import br.com.escola.gestaoescolar.dominio.Curso;
import br.com.escola.gestaoescolar.dominio.Periodo;
import br.com.escola.gestaoescolar.dominio.Turma;

import java.time.LocalDate;

public class Testes {

    public static void main(String[] args) {
        Curso ingles = new Curso("Ingles basico", 100);
        Turma t1 = new Turma("T-01",
                ingles,
                LocalDate.of(2023, 11, 07),
                LocalDate.of(2023, 12, 07),
                Periodo.SABADOS);

        System.out.println(t1.getPeriodo().getNome());
    }

}
