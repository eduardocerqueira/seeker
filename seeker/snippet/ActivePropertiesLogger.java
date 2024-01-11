//date: 2024-01-11T16:43:44Z
//url: https://api.github.com/gists/3133af95ef08e0cb26382b1e3b864f6d
//owner: https://api.github.com/users/Marcel510

package de.check24.kfzif.sofortevb;

import org.springframework.boot.env.OriginTrackedMapPropertySource;
import org.springframework.context.event.ContextRefreshedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.stream.Stream;

@Component
public class ActivePropertiesLogger {

    @EventListener
    public void printProperties(ContextRefreshedEvent contextRefreshedEvent) {
        System.out.println("************************* ACTIVE PROPERTIES *************************");

        ((ConfigurableEnvironment) contextRefreshedEvent.getApplicationContext().getEnvironment())
            .getPropertySources()
            .stream()
            .flatMap(propertySource -> propertySource instanceof OriginTrackedMapPropertySource originTrackedMapPropertySource
                ? Stream.of(originTrackedMapPropertySource)
                : Stream.empty())
            // Convert each PropertySource to its properties Set
            .map(propertySource -> propertySource.getSource().entrySet())
            .flatMap(Collection::stream)
            // Print properties within each Set
            .forEach(property -> System.out.println(property.getKey() + "=" + property.getValue()));

        System.out.println("*********************************************************************");
    }

}
