//date: 2022-09-06T17:04:54Z
//url: https://api.github.com/gists/e5cde6f2ffcd65ee4e31ac7c723a2041
//owner: https://api.github.com/users/thiagonuneszup

public class AutorResponse {

    private String nome;
    private String email;

    public AutorResponse(String nome, String email) {
        this.nome = nome;
        this.email = email;
    }

    public String getNome() {
        return nome;
    }

    public String getEmail() {
        return email;
    }
}
