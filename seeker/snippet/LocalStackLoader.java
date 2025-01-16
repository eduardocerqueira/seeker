//date: 2025-01-16T17:10:14Z
//url: https://api.github.com/gists/13cd8e0f6f4244714f6a74e572e2413e
//owner: https://api.github.com/users/GVabal

import io.micronaut.context.event.ShutdownEvent;
import io.micronaut.context.event.StartupEvent;
import io.micronaut.runtime.event.annotation.EventListener;
import jakarta.inject.Singleton;

import java.io.IOException;

@Singleton
public class LocalStackLoader {

    @EventListener
    public void startDockerCompose(StartupEvent startupEvent) throws IOException {
        System.out.println("Start docker compose");
        Runtime.getRuntime().exec(new String[]{"docker", "compose", "up", "-d"}).inputReader().lines().forEach(System.out::println);
    }

    @EventListener
    public void stopDockerCompose(ShutdownEvent shutdownEvent) throws IOException {
        System.out.println("Stop docker compose");
        Runtime.getRuntime().exec(new String[]{"docker", "compose", "down"}).inputReader().lines().forEach(System.out::println);
    }
}
