//date: 2022-07-25T17:18:49Z
//url: https://api.github.com/gists/c1b8a5470cdf2443e5f3e08231528a69
//owner: https://api.github.com/users/danyllosoareszup

@Entity
public class Zupper {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String nome;

    @NotNull
    private LocalDate dataAdmicao;

    @Column(nullable = false)
    private String email;

    @ManyToMany(mappedBy = "zuppers")
    private Set<Palestra> palestras = new HashSet<>();

    @Deprecated
    public Zupper() {
    }

    public Zupper(String nome, LocalDate dataAdmicao, String email) {
        this.nome = nome;
        this.dataAdmicao = dataAdmicao;
        this.email = email;
    }

    public Long getId() {
        return id;
    }

    public void adicionarPalestra(Palestra palestra) {
        this.palestras.add(palestra);
    }

    public void removerPalestra(Palestra palestra) {
        this.palestras.remove(palestra);
    }
}