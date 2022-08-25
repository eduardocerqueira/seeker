//date: 2022-08-25T15:29:08Z
//url: https://api.github.com/gists/f06c32c86f5b8a886cdf5bc2dfec93f7
//owner: https://api.github.com/users/thiagonuneszup

class NumeroPrimoTest {

    private NumeroPrimo numeroPrimo = new NumeroPrimo();


    @ParameterizedTest(
            name = "{index} => valor={0}"
    )
    @DisplayName("nao deve ser primo, mc/dc primero if")
    @MethodSource("primeiroIfNaoPrimoProvider")
    void test(int valor) {
        assertFalse(numeroPrimo.primo(valor));
    }

    @Test
    @DisplayName("deve ser primo, mc/dc segundo e terceiro if")
    void test1() {
        assertTrue(numeroPrimo.primo(5));
    }

    @Test
    @DisplayName("deve ser primo, mc/dc ultimo if caso de teste 2")
    void test2() {

        assertFalse(numeroPrimo.primo(4,2,1));
    }

    private static Stream<Arguments> primeiroIfNaoPrimoProvider() {
        return Stream.of(
                Arguments.of(-1),
                Arguments.of(0),
                Arguments.of(1)
        );
    }

}