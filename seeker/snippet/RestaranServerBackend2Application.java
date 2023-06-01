//date: 2023-06-01T17:05:10Z
//url: https://api.github.com/gists/63131d6035a54d88c1132fbc6bb77bbb
//owner: https://api.github.com/users/shavkatnazarov

package it.ul.restaranserverbackend2;

import it.ul.restaranserverbackend2.config.InitConfig;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RestaranServerBackend2Application {

    public static void main(String[] args) {
        if (InitConfig.isStart()) {
            SpringApplication.run(RestaranServerBackend2Application.class, args);
        }
    }

}
