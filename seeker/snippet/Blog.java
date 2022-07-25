//date: 2022-07-25T16:54:17Z
//url: https://api.github.com/gists/5de43f3fb41a1921d58318c2d7084f93
//owner: https://api.github.com/users/joaosazup

@Entity
public class Blog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    @Column(nullable = false)
    private String nome;

    @OneToMany(cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Artigo> artigos = new ArrayList<>();

    private LocalDateTime dateEHoraCriacao = LocalDateTime.now();

    /**
    * @deprecated Contrutor de uso exclusivo do hibernate
    */
    @Deprecated
    public Blog() {
    }

    public Blog(@NotBlank String nome) {
        this.nome = nome;
    }

    public Long getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public LocalDateTime getDateEHoraCriacao() {
        return dateEHoraCriacao;
    }

    public List<Artigo> getArtigos() {
        return artigos;
    }

    public void adiciona(Artigo artigo) {
        artigos.add(artigo);
    }

}
