//date: 2021-11-11T17:13:21Z
//url: https://api.github.com/gists/c304ddbea96638fbb0ee852145b4efa5
//owner: https://api.github.com/users/schroedermatt

import org.springframework.kafka.test.EmbeddedKafkaBroker;

public class EmbeddedKafkaTestResource implements QuarkusTestResourceLifecycleManager {

    public EmbeddedKafkaBroker embeddedBroker;

    /**
     * @return A map of system properties that should be set for the running test
     */
    @Override
    public Map<String, String> start() {
        embeddedBroker = new EmbeddedKafkaBroker(1);

        Map<String,String> brokerProperties = new HashMap<>();

        // add any specific broker properties here (auto topic creation, default topic settings, etc)
        // brokerProperties.put("key","value");

        embeddedBroker.brokerProperties(brokerProperties);

        // The system property with this name is set to the list of broker addresses
        embeddedBroker.brokerListProperty("kafka.bootstrap.servers");

        // initialize the broker
        embeddedBroker.afterPropertiesSet();

        // map of system properties that should be set for the running test
        Map<String, String> props = new HashMap<>();
        props.put("kafka.bootstrap.servers", embeddedBroker.getBrokersAsString());
        props.put("quarkus.kafka-streams.bootstrap-servers", embeddedBroker.getBrokersAsString());

        return props;
    }

    @Override
    public void stop() {
        if (embeddedKafkaBroker != null) {
            embeddedKafkaBroker.destroy();
        }
    }
}