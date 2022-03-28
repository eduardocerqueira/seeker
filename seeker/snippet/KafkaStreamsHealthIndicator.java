//date: 2022-03-28T17:06:03Z
//url: https://api.github.com/gists/259cab2bea73cb1f0b114bef620dcbdb
//owner: https://api.github.com/users/MarioCroSite

import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.KafkaStreams.State;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.kafka.core.StreamsBuilderFactoryBean;
import org.springframework.stereotype.Component;

@Component
public class KafkaStreamsHealthIndicator implements HealthIndicator {

  private final StreamsBuilderFactoryBean defaultKafkaStreamsBuilder;

  @Autowired
  public KafkaStreamsHealthIndicator(
      @Qualifier("defaultKafkaStreamsBuilder") StreamsBuilderFactoryBean defaultKafkaStreamsBuilder) {
    this.defaultKafkaStreamsBuilder = defaultKafkaStreamsBuilder;
  }

  @Override
  public Health health() {
    final KafkaStreams kafkaStreams = defaultKafkaStreamsBuilder.getKafkaStreams();

    if (kafkaStreams == null) {
      return Health.down().build();
    }
    else {
      final State state = kafkaStreams.state();
      if (state == State.RUNNING) {
        return Health.up().build();
      }
      else {
        return Health.down()
            .withDetail("state", state)
            .build();
      }
    }
  }
}
