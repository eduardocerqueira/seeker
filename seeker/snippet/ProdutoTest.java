//date: 2022-08-25T15:10:17Z
//url: https://api.github.com/gists/342fde9acd3f1a682bd23aa40a25c71a
//owner: https://api.github.com/users/thiagonuneszup

class ProdutoTest {

    private Usuario thiago;
    protected Usuario daniel;
    private BigDecimal desconto;
    private Produto produto1;
    private Produto produto2;

    @BeforeEach
    void setUp() {
        this.thiago = new Usuario();
        this.daniel = new Usuario();
        this.desconto = new BigDecimal("0.1");
        this.produto1 = new Produto("Sanduíche", BigDecimal.TEN);
        this.produto2 = new Produto("x-bacon", BigDecimal.TEN);
    }

    @Test
    @DisplayName("deve gerar uma compra com desconto de um cumpom")
    void test() {

        CupomDesconto cupomDesconto = new CupomDesconto(
                thiago,
                produto1,
                desconto,
                LocalDateTime.now().plusDays(2)
        );

        Compra comprar = produto1.comprar(cupomDesconto, thiago);

        assertEquals(
                BigDecimal.TEN.multiply(BigDecimal.ONE.subtract(desconto)),
                comprar.getValor()
        );
    }

    @Test
    @DisplayName("nao deve gerar uma compra com desconto caso a data do cumpom seja expirada")
    void test2() {

        CupomDesconto cupomDesconto = new CupomDesconto(
                thiago,
                produto1,
                desconto,
                LocalDateTime.now().minusDays(2)
        );

        Compra comprar = produto1.comprar(cupomDesconto, thiago);

        assertEquals(
                BigDecimal.TEN,
                comprar.getValor()
        );

    }


    @Test
    @DisplayName("nao deve gerar uma compra com desconto caso o cumpom nao pertenca ao usuario")
    void test3() {

        CupomDesconto cupomDesconto = new CupomDesconto(
                daniel,
                produto1,
                desconto,
                LocalDateTime.now()
        );

        Compra comprar = produto1.comprar(cupomDesconto, thiago);

        assertEquals(
                BigDecimal.TEN,
                comprar.getValor()
        );
    }

    @Test
    @DisplayName("nao deve gerar uma compra com desconto caso o cumpom nao pertenca ao produto")
    void test4() {

        CupomDesconto cupomDesconto = new CupomDesconto(
                daniel,
                produto2,
                desconto,
                LocalDateTime.now()
        );

        Compra comprar = produto1.comprar(cupomDesconto,thiago);

        assertEquals(
                BigDecimal.TEN,
                comprar.getValor()
        );
    }
}
