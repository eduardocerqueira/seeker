//date: 2022-08-01T17:19:46Z
//url: https://api.github.com/gists/06a05dcc3e8f9d4a6955f0d428106ddf
//owner: https://api.github.com/users/danielmotazup

@Entity
public class Reserva {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    private Quarto quarto;

    @JsonFormat(pattern = "dd/MM/yyyy")
    private LocalDate checkin;

    @JsonFormat(pattern = "dd/MM/yyyy")
    private LocalDate checkout;

    @JsonFormat(pattern = "dd/MM/yyyy HH:mm:ss")
    private LocalDateTime dataRegistro = LocalDateTime.now();

    @Deprecated
    public Reserva() {
    }

    public Reserva(LocalDate checkin, LocalDate checkout) {
        this.checkin = checkin;
        this.checkout = checkout;
    }

    public Long getId() {
        return id;
    }

    public void adicionaQuarto(Quarto quarto){
        this.quarto = quarto;
    }
}
