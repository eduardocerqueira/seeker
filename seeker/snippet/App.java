//date: 2025-12-30T16:38:50Z
//url: https://api.github.com/gists/c18ebfa6395db858403bdaa77063d554
//owner: https://api.github.com/users/msgilligan

///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21
//JAVAC_OPTS -proc:full
//DEPS io.micronaut.platform:micronaut-platform:4.10.3@pom
//DEPS io.micronaut:micronaut-http-server-netty
//DEPS io.micronaut:micronaut-inject-java
//DEPS io.micronaut:micronaut-jackson-databind
//DEPS org.slf4j:slf4j-simple
package app;

import io.micronaut.http.annotation.*;
import io.micronaut.http.MediaType;
import io.micronaut.runtime.Micronaut;

public class App {

    public static void main(String... args) {
        Micronaut.run(args);
    }
}

@Controller("/hello")
class HelloController {
    @Get
    @Produces(MediaType.TEXT_PLAIN)
    String hello() {
        return "Hello world";
    }
}
