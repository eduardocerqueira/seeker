//date: 2022-03-28T17:07:12Z
//url: https://api.github.com/gists/658323290e72130874037d4f2fda065b
//owner: https://api.github.com/users/MarioCroSite

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.binder.kafka.KafkaStreamsMetrics;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.StreamsBuilderFactoryBeanCustomizer;

/**
 * Binds a micrometer {@link KafkaStreamsMetrics} to the meter registry
 * for each Kafka stream created by Spring.
 */
@Configuration
public class KafkaStreamsMetricsConfig {

  @Bean
  public StreamsBuilderFactoryBeanCustomizer streamsBuilderCustomizer(MeterRegistry meterRegistry) {
    return streamsBuilderFactoryBean ->
        streamsBuilderFactoryBean.setKafkaStreamsCustomizer(
            kafkaStreams ->
                new KafkaStreamsMetrics(kafkaStreams).bindTo(meterRegistry)
        );
  }

}
