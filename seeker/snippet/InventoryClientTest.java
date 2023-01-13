//date: 2023-01-13T16:43:13Z
//url: https://api.github.com/gists/ca21e51cd4568f89d1bc09ad1d0a6b42
//owner: https://api.github.com/users/bjerat

class InventoryClientTest {

    private InventoryClient cut;

    public static MockWebServer mockBackEnd;

    @BeforeAll
    public static void setUp() throws IOException {
        mockBackEnd = new MockWebServer();
        mockBackEnd.start();
    }

    @AfterAll
    public static void tearDown() throws IOException {
        mockBackEnd.shutdown();
    }

    @BeforeEach
    public void before() {
        cut = new InventoryClient(new InventoryClientProperties()
                .setUrl(mockBackEnd.url("/").toString())
                .setRetry(new RetryBackoffProperties()
                        .setMaxAttempts(2)
                        .setBackoffDuration(Duration.ofMillis(10000))));
    }

    ...
}