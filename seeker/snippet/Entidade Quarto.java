//date: 2022-08-01T17:18:24Z
//url: https://api.github.com/gists/04b7e9fa08a336280e82b0790655d353
//owner: https://api.github.com/users/danielmotazup

@Entity
public class Quarto {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 200)
    private String descricao;

    private BigDecimal diaria;

    private TipoCama cama;

    private boolean ativo;

    @OneToMany(cascade = CascadeType.ALL, mappedBy = "quarto")
    private List<Reserva> reservas = new ArrayList<>();


    @Version
    private int versao;

    public Long getId() {
        return id;
    }

    public boolean isAtivo() {
        return ativo;
    }

    public void setAtivo(boolean ativo) {
        this.ativo = ativo;
    }

    @Deprecated
    public Quarto() {
    }

    public Quarto(String descricao, BigDecimal diaria, TipoCama cama) {
        this.descricao = descricao;
        this.diaria = diaria;
        this.cama = cama;
    }

    public void adicionarReserva(Reserva reserva){
        this.reservas.add(reserva);
        reserva.adicionaQuarto(this);
        this.setAtivo(true);
    }
}