//date: 2022-07-30T18:59:25Z
//url: https://api.github.com/gists/eeff30918bd4bb01cd6f9770f8383639
//owner: https://api.github.com/users/danyllosoareszup

public class HabilidadesResponse {

    private String nome;

    private String nivel;

    public HabilidadesResponse(Habilidade habilidade) {
        this.nome = habilidade.getNome();
        this.nivel = String.valueOf(habilidade.getNivel());
    }

    @Deprecated
    public HabilidadesResponse() {
    }

    public String getNivel() {
        return nivel;
    }

    public String getNome() {
        return nome;
    }
}