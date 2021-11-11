//date: 2021-11-11T17:09:55Z
//url: https://api.github.com/gists/589c87e6d9766308b7fd046847ed1630
//owner: https://api.github.com/users/schroedermatt

public class ContainerKafkaTestResource implements QuarkusTestResourceLifecycleManager {

    private static final KafkaContainer kafka = new KafkaContainer();

    /**
     * @return A map of system properties that should be set for the running tests
     */
    @Override
    public Map<String, String> start() {
        kafka.start();

        Map<String, String> systemProperties = new HashMap<>();
        systemProperties.put("kafka.bootstrap.servers", kafka.getBootstrapServers());
        systemProperties.put("quarkus.kafka-streams.bootstrap-servers", kafka.getBootstrapServers());

        return systemProperties;
    }

    @Override
    public void stop() {
        kafka.close();
    }
}