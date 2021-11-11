//date: 2021-11-11T17:10:55Z
//url: https://api.github.com/gists/601bb1cea69b9a7ca8260f48fbb55964
//owner: https://api.github.com/users/schroedermatt

@QuarkusTest
@QuarkusTestResource(ContainerKafkaTestResource.class)
@Stereotype
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@DisabledIfSystemProperty(named = "container-enabled", matches = "false")
public @interface QuarkusContainerKafkaTest {}