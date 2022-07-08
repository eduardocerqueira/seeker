//date: 2022-07-08T17:11:09Z
//url: https://api.github.com/gists/8f629d031ff0d1b23d6cdf8ec5648f83
//owner: https://api.github.com/users/ravi-shankar-v

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ContextConfiguration(initializers = AbstractIntegrationTest.Initializer.class)
@Testcontainers
@ActiveProfiles({"integration-test"})
class AbstractIntegrationTest {

    protected static final int ServerPort = 8090;
    @Container
    private static final CosmosDBEmulatorContainer emulator = new CosmosDBEmulatorContainer(
            DockerImageName.parse("mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest")
    );
    @TempDir
    private static Path tempFolder;

    static class Initializer
            implements ApplicationContextInitializer<ConfigurableApplicationContext> {

        @SneakyThrows
        @Override
        public void initialize(ConfigurableApplicationContext context) {
            Path keyStoreFile = tempFolder.resolve("azure-cosmos-emulator.keystore");
            KeyStore keyStore = emulator.buildNewKeyStore();
            keyStore.store(new FileOutputStream(keyStoreFile.toFile()), emulator.getEmulatorKey().toCharArray());

            System.setProperty("javax.net.ssl.trustStore", keyStoreFile.toString());
            System.setProperty("javax.net.ssl.trustStorePassword", emulator.getEmulatorKey());
            System.setProperty("javax.net.ssl.trustStoreType", "PKCS12");

            TestPropertyValues values = TestPropertyValues.of("azure.cosmosdb.uri=" + emulator.getEmulatorEndpoint(),
                    "azure.cosmosdb.key=" + emulator.getEmulatorKey(),
                    "server.port=" + ServerPort);

            values.applyTo(context);
        }
    }

    @TestConfiguration
    static class DemoConfiguration {

        @Value("${azure.cosmosdb.uri}")
        private String testContainerURI;

        @Value("${azure.cosmosdb.key}")
        private String testContainerKey;

        @Bean
        public CosmosClientBuilder cosmosClient() {
            return new CosmosClientBuilder()
                    .endpoint(testContainerURI)
                    .key(testContainerKey)
                    .gatewayMode()
                    .endpointDiscoveryEnabled(false);
        }
    }

}