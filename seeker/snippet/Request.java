//date: 2022-07-25T16:58:39Z
//url: https://api.github.com/gists/1ce3240f076699c75a60789aaff8a35e
//owner: https://api.github.com/users/joaosazup

public class CadastrarArtigoRequest {
    @NotBlank
    @Length(max = 200)
    private String titulo;

    @NotBlank
    @Length(max = 10000)
    private String corpo;

    @NotNull
    private TipoArtigoRequest tipo;

    public Artigo toArtigo() {
       return new Artigo(titulo, corpo, tipo.toTipoArtigo()); 
    }

    public String getTitulo() {
        return titulo;
    }

    public String getCorpo() {
        return corpo;
    }

    public TipoArtigoRequest getTipo() {
        return tipo;
    }
}