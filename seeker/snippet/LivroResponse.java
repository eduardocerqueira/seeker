//date: 2022-09-06T17:04:54Z
//url: https://api.github.com/gists/e5cde6f2ffcd65ee4e31ac7c723a2041
//owner: https://api.github.com/users/thiagonuneszup

public class LivroResponse {

    private Long id;
    private String nome;
    private String descricao;
    private String isbn;
    private LocalDate publicadoEm;
    private AutorResponse autor;



    public LivroResponse(LivroResponseApiExterna livroResponse, AutorResponse autorResponse) {
        this.id = livroResponse.getId();
        this.nome = livroResponse.getNome();
        this.descricao = livroResponse.getDescricao();
        this.isbn = livroResponse.getIsbn();
        this.publicadoEm = livroResponse.getPublicadoEm();
        this.autor = new AutorResponse(autorResponse.getNome(),autorResponse.getEmail());
    }

    public Long getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public String getDescricao() {
        return descricao;
    }

    public String getIsbn() {
        return isbn;
    }

    public LocalDate getPublicadoEm() {
        return publicadoEm;
    }

    public AutorResponse getAutor() {
        return autor;
    }
}
