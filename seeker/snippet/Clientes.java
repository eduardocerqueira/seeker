//date: 2022-04-19T16:52:04Z
//url: https://api.github.com/gists/3555795d34fed4d2382c57036f8d03f5
//owner: https://api.github.com/users/DiogoLuizDeAquino

package Implementando_Conta_P;

public class Clientes {
    private String nome;
    private String cpf;

    public Clientes(String nome, String cpf) {
        this.nome = nome;
        this.cpf = cpf;
    }

    public String getNome() {
        return this.nome;
    }

    public String getCpf() {
        return this.cpf;
    }

}
