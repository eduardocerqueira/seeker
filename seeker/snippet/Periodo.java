//date: 2023-10-20T17:01:57Z
//url: https://api.github.com/gists/485e40f11fc8d30b12ac5a22414fa8c0
//owner: https://api.github.com/users/canaldojavao

package br.com.escola.gestaoescolar.dominio;

public enum Periodo {

    MATUTINO("Matutino"),
    VESPERTINO("Vespertino"),
    NOTURNO("Noturno"),
    SABADOS("Sábados");

    private final String nome;

    private Periodo(String nome) {
        this.nome = nome;
    }

    public String getNome() {
        return nome;
    }
}
