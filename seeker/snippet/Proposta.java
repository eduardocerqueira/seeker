//date: 2022-09-27T17:08:32Z
//url: https://api.github.com/gists/17d74ff8352c97715e1a76a41f140f08
//owner: https://api.github.com/users/danielmotazup

@Entity
public class Proposta {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String nome;

    private String cpf;

    @ManyToOne(cascade = CascadeType.ALL)
    private Endereco endereco;

    public Proposta(String nome, String cpf, Endereco endereco) {
        this.nome = nome;
        this.cpf = cpf;
        this.endereco = endereco;
    }

    public Long getId() {
        return id;
    }
}