//date: 2022-07-20T16:59:29Z
//url: https://api.github.com/gists/36994bce35c922dc15e320b0ac85c4f0
//owner: https://api.github.com/users/henriquesousazup

@Entity
public class Zupper {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String nome;

    @Column(nullable = false)
    private String email;

    @Column(nullable = false)
    @Enumerated(value = EnumType.STRING)
    private CargoEnum cargo;

    @OneToMany(cascade = CascadeType.PERSIST, orphanRemoval = true, mappedBy = "zupper")
    private List<Endereco> enderecos = new ArrayList<>();

    @Deprecated
    /**
     * @deprecated para uso exclusivo do hibernate
     */
    public Zupper() {
    }

    public Zupper(String nome, String email, CargoEnum cargo) {
        this.nome = nome;
        this.email = email;
        this.cargo = cargo;
    }

    public Long getId() {
        return id;
    }

    public void adiciona(Endereco endereco) {
        endereco.setZupper(this);
        this.enderecos.add(endereco);
    }
}