//date: 2025-04-23T17:09:31Z
//url: https://api.github.com/gists/438c3f8e9349c02df9a3aabc57e5a436
//owner: https://api.github.com/users/luanfranciscojr

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class MeuControladorTest {

    @Autowired
    private TestRestTemplate restTemplate;

    @Test
    public void testGetMensagem() {
        String resposta = restTemplate.getForObject("/api/mensagem", String.class);
        assertEquals("Olá, Spring Boot!", resposta);
    }
}