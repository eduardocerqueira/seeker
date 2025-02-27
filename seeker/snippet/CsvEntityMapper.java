//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Profile(value = { "dev", "seed" })
@Component
public @interface CsvEntityMapper {
    String value();
}
