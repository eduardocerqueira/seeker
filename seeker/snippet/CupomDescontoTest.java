//date: 2022-08-25T15:11:29Z
//url: https://api.github.com/gists/459f4d6d45d9a42d266a10f459e21b76
//owner: https://api.github.com/users/thiagonuneszup

class CupomDescontoTest {

    private Usuario usuario;
    private CupomDesconto cupomDesconto;
    private Produto produto;
    private BigDecimal porcentagem;
    private final LocalDateTime HOJE = LocalDateTime.now();

    @BeforeEach
    void setUp() {
        this.usuario = new Usuario();
        this.produto = new Produto("x-tudo",new BigDecimal("25"));
        this.porcentagem = new BigDecimal("0.1");
    }

    @Test
    @DisplayName("o cupom nao deve ser valido caso a data atual seja maior q a validade")
    void test1() {
        this.cupomDesconto = new CupomDesconto(usuario,produto,porcentagem,HOJE.minusDays(2));

        assertFalse(cupomDesconto.isValido());
    }

    @Test
    @DisplayName("o cupom deve ser valido caso a data atual seja igual a data validade")
    void test2() {

        this.cupomDesconto = new CupomDesconto(usuario,produto,porcentagem,LocalDateTime.now());
        assertTrue(cupomDesconto.isValido());



    }

    @Test
    @DisplayName("o cupom deve ser valido caso a data atual seja menor que a data de validade")
    void test3(){
        this.cupomDesconto = new CupomDesconto(usuario,produto,porcentagem,HOJE.plusDays(1));

        assertTrue(cupomDesconto.isValido());
    }
