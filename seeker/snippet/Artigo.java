//date: 2022-07-25T16:56:36Z
//url: https://api.github.com/gists/6e0423e93d1f0f0bd5e68b5d3ad547a3
//owner: https://api.github.com/users/joaosazup

@Entity
public class Artigo {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @NotBlank
    @Length(max = 200)
    @Column(nullable = false)
    private String titulo;

    @NotBlank
    @Length(max = 10000)
    @Column(nullable = false)
    private String corpo;

    @NotNull
    @Enumerated(EnumType.STRING)
    private TipoArtigo tipo;

    /**
    * @deprecated Contrutor de uso exclusivo do hibernate
    */
    @Deprecated
    public Artigo() {
    }

    public Artigo(@NotBlank @Length(max = 200) String titulo, @NotBlank @Length(max = 10000) String corpo,
            @NotNull TipoArtigo tipo) {
        this.titulo = titulo;
        this.corpo = corpo;
        this.tipo = tipo;
    }

    public Long getId() {
        return id;
    }

    public String getTitulo() {
        return titulo;
    }

    public String getCorpo() {
        return corpo;
    }

    public TipoArtigo getTipo() {
        return tipo;
    }
}
