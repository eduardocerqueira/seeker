//date: 2025-09-22T16:58:24Z
//url: https://api.github.com/gists/398f8eea2769495d395e83d31c7a0194
//owner: https://api.github.com/users/tuxmonteiro

package com.maoudia;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@EnableConfigurationProperties(AppProperties.class)
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}