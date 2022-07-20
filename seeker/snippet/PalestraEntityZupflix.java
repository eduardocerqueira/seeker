//date: 2022-07-20T17:15:15Z
//url: https://api.github.com/gists/ded592e39785e4066e37e68843a88899
//owner: https://api.github.com/users/joaobragazup

@Entity
public class Palestra {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String titulo;

    @Column(nullable = false)
    private String tema;

    @Column(nullable = false)
    //private LocalDate tempoPalestra = LocalDate.now();
    private Integer tempoPalestra;

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    private TipoExibicao tipoExibicao;

    @Column(nullable = false)
    private LocalDate horaExibicao = LocalDate.now();

    @JoinTable(
            name = "zupper_palestra",
            joinColumns = @JoinColumn(name = "zupper_id"),
            inverseJoinColumns = @JoinColumn(name = "palestra_id")
    )
    @ManyToMany(cascade = {CascadeType.MERGE,CascadeType.PERSIST})
    private Set<Zupper> zuppers = new HashSet<>();

    public Palestra(String titulo, String tema, Integer tempoPalestra, TipoExibicao tipoExibicao, LocalDate horaExibicao) {
        this.titulo = titulo;
        this.tema = tema;
        this.tempoPalestra = tempoPalestra;
        this.tipoExibicao = tipoExibicao;
        this.horaExibicao = horaExibicao;
    }

    @Deprecated
    public Palestra() {
    }

    public Long getId() {
        return id;
    }

    public void adiciona(Set<Zupper> zupper){
        this.zuppers.addAll(zupper);
    }
}
