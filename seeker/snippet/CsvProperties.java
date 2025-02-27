//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Profile;

import java.util.List;

@Profile(value = { "dev", "seed" })
@ConfigurationProperties(prefix = "app.seed.csv")
public record CsvProperties(String basePath, List<String> files) {
}