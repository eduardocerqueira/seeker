//date: 2022-09-06T17:04:54Z
//url: https://api.github.com/gists/e5cde6f2ffcd65ee4e31ac7c723a2041
//owner: https://api.github.com/users/thiagonuneszup

public class LivroResponseApiExterna {

    private Long id;
    private String nome;
    private String descricao;
    private String isbn;
    private Long autorId;
    private LocalDate publicadoEm;
    private LocalDateTime criadoEm;

    public LivroResponseApiExterna(Long id, String nome, String descricao, String isbn, Long autorId, LocalDate publicadoEm, LocalDateTime criadoEm) {
        this.id = id;
        this.nome = nome;
        this.descricao = descricao;
        this.isbn = isbn;
        this.autorId = autorId;
        this.publicadoEm = publicadoEm;
        this.criadoEm = criadoEm;
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

    public Long getAutorId() {
        return autorId;
    }

    public LocalDate getPublicadoEm() {
        return publicadoEm;
    }

    public LocalDateTime getCriadoEm() {
        return criadoEm;
    }
}
