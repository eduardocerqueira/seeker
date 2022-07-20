//date: 2022-07-20T16:59:50Z
//url: https://api.github.com/gists/bc276d6ccd7cde39329d09efddfaa945
//owner: https://api.github.com/users/joaobragazup

@Entity
public class Zupper {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String nome;

    @Column(nullable = false)
    private LocalDate dataAdimissao = LocalDate.now();

    @Column(nullable = false)
    private String email;

    public Zupper(String nome, LocalDate dataAdimissao, String email) {
        this.nome = nome;
        this.dataAdimissao = dataAdimissao;
        this.email = email;
    }

    @Deprecated
    public Zupper() {
    }

    public Long getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public LocalDate getDataAdimissao() {
        return dataAdimissao;
    }

    public String getEmail() {
        return email;
    }
}
