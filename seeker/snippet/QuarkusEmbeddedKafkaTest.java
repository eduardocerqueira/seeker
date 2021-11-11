//date: 2021-11-11T17:14:07Z
//url: https://api.github.com/gists/9de4fecaf70de431c01efcc9ada6cd2e
//owner: https://api.github.com/users/schroedermatt

@QuarkusTest
@QuarkusTestResource(EmbeddedKafkaTestResource.class)
@Stereotype
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface QuarkusEmbeddedKafkaTest {}