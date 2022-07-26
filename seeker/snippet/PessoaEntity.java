//date: 2022-07-26T17:00:52Z
//url: https://api.github.com/gists/82d4650e2c46662c9278a2a268091c19
//owner: https://api.github.com/users/joaobragazup

@Entity
public class Pessoa {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String nome;
    private String cpf;
    private String apelido;
    private LocalDate dataNascimento;

    public Pessoa(String nome, String cpf, String apelido, LocalDate dataNascimento) {
        this.nome = nome;
        this.cpf = cpf;
        this.apelido = apelido;
        this.dataNascimento = dataNascimento;
    }

    public Integer calculaIdade(){
        LocalDate hoje = LocalDate.now();

        Integer idade;

        idade = hoje.getYear() - dataNascimento.getYear() ;

        return idade;
    }

    /**
     * @deprecated
     */
    @Deprecated
    public Pessoa() {
    }

    public Long getId() {
        return id;
    }
}