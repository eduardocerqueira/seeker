//date: 2022-08-25T15:25:11Z
//url: https://api.github.com/gists/38ab4a3094d49f3dd794be78d191e055
//owner: https://api.github.com/users/thiagonuneszup

class CartaoTest {

    private Cartao cartao;
    private List<Fatura> faturas;
    private YearMonth MES_ATUAL = YearMonth.now();
    private YearMonth PROXIMO_MES = MES_ATUAL.plusMonths(1);

    private YearMonth DOIS_MESES_DE_HOJE = PROXIMO_MES.plusMonths(1);


    @Test
    @DisplayName("deve aprovar uma compra caso o tenha limite disponivel e codigo de seguranca e senha sejam validos")
    void test() {
        String codigoSeguranca = "234";
        String senha = "**********"

        this.cartao = new Cartao(
                "12321",
                "Jordi h",
                codigoSeguranca,
                senha,
                new BigDecimal("130"),
                LocalDateTime.now().plusYears(2)
        );

        faturas = criaFaturas(this.cartao);
        cartao.adicionar(faturas);

        assertTrue(cartao.isAprovado(new Gasto(BigDecimal.TEN), "234", "1234"));
    }

    @Test
    @DisplayName("nao deve aprovar caso a senha esteja incorreta")
    void test1() {
        String codigoSeguranca = "234";


        this.cartao = new Cartao(
                "12321",
                "Jordi h",
                codigoSeguranca,
                "1234",
                new BigDecimal("130"),
                LocalDateTime.now().plusYears(2)
        );

        faturas = criaFaturas(this.cartao);
        cartao.adicionar(faturas);

        assertFalse(cartao.isAprovado(new Gasto(BigDecimal.TEN), codigoSeguranca, "12345"));
    }


    @Test
    @DisplayName("nao deve aprovar caso o codigo de segurança esteja incorreta")
    void test2() {

        this.cartao = new Cartao(
                "12321",
                "Jordi h",
                "124",
                "1234",
                new BigDecimal("130"),
                LocalDateTime.now().plusYears(2)
        );

        faturas = criaFaturas(this.cartao);
        cartao.adicionar(faturas);

        assertFalse(cartao.isAprovado(new Gasto(BigDecimal.TEN), "123", "1234"));
    }

    @Test
    @DisplayName("nao deve aprovar caso o limite seja insuficiente")
    void test3() {

        this.cartao = new Cartao(
                "12321",
                "Jordi h",
                "124",
                "1234",
                new BigDecimal("120"),
                LocalDateTime.now().plusYears(2)
        );

        faturas = criaFaturas(this.cartao);
        cartao.adicionar(faturas);

        assertFalse(cartao.isAprovado(new Gasto(BigDecimal.TEN), "124", "1234"));
    }

    private List<Fatura> criaFaturas(Cartao cartao) {
        return List.of(
                new Fatura(
                        cartao,
                        MES_ATUAL,
                        List.of(
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN)
                        )
                ),
                new Fatura(
                        cartao,
                        PROXIMO_MES,
                        List.of(
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN)
                        )
                ),
                new Fatura(
                        cartao,
                        DOIS_MESES_DE_HOJE,
                        List.of(
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN),
                                new Gasto(BigDecimal.TEN)
                        )
                )
        );
    }

}