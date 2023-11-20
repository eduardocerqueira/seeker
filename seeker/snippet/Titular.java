//date: 2023-11-20T16:32:02Z
//url: https://api.github.com/gists/db9930c6b867fd70d067b9b8b0612907
//owner: https://api.github.com/users/canaldojavao

package br.com.empresa.banco.titular;

public class Titular {

    private String nome;
    private String cpf;
    private String email;

    public Titular(String nome, String cpf, String email) {
        this.nome = nome;
        this.cpf = cpf;
        this.email = email;
    }

    public String getNome() {
        return nome;
    }

    public String getCpf() {
        return cpf;
    }

    public String getEmail() {
        return email;
    }

}
