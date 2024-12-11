//date: 2024-12-11T17:09:27Z
//url: https://api.github.com/gists/9ad4c1f1434308d2b438e611b69fe48f
//owner: https://api.github.com/users/valdemarjuniorr

package br.com.valdemarjr;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.localstack.LocalStackContainer;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.utility.MountableFile;


/**
 * Class should be extended by @Controller's classes. This class is responsible for abstract
 * integration tests configurations.
 */
@AutoConfigureMockMvc
@ActiveProfiles("test")
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        classes = Application.class)
public abstract class AbstractIntegrationTest {

    protected static final String LOCALHOST = "http://localhost";

    @LocalServerPort
    private int port;

    @Autowired
    protected TestRestTemplate restTemplate;
    
    /**
     * Initialize Localstack container for integration tests
     */
    static LocalStackContainer localstack = new LocalStackContainer(DockerImageName.parse("localstack/localstack:3.5.0"))
            .withCopyFileToContainer(MountableFile.forClasspathResource("./scripts/init-aws.sh"), "/etc/localstack/init/ready.d/init-aws.sh")
            .withServices(LocalStackContainer.Service.SNS, LocalStackContainer.Service.SQS);
    
    @BeforeAll
    static void startPostgresContainer() {
        System.setProperty("aws.accessKeyId", localstack.getAccessKey());
        System.setProperty("aws.secretAccessKey", localstack.getSecretKey());
        System.setProperty("aws.region", localstack.getSecretKey());
        localstack.start();
    }

    @AfterAll
    static void stopPostgresContainer() {
        localstack.stop();
    }

    /**
     * Configure properties dynamically defined in application.yml file
     */
    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.cloud.aws.region.static", localstack::getRegion);
        registry.add("spring.cloud.aws.credentials.access-key", localstack::getAccessKey);
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"r "**********"e "**********"g "**********"i "**********"s "**********"t "**********"r "**********"y "**********". "**********"a "**********"d "**********"d "**********"( "**********"" "**********"s "**********"p "**********"r "**********"i "**********"n "**********"g "**********". "**********"c "**********"l "**********"o "**********"u "**********"d "**********". "**********"a "**********"w "**********"s "**********". "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********". "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"- "**********"k "**********"e "**********"y "**********"" "**********", "**********"  "**********"l "**********"o "**********"c "**********"a "**********"l "**********"s "**********"t "**********"a "**********"c "**********"k "**********": "**********": "**********"g "**********"e "**********"t "**********"S "**********"e "**********"c "**********"r "**********"e "**********"t "**********"K "**********"e "**********"y "**********") "**********"; "**********"
        registry.add("spring.cloud.aws.endpoint.uri", () -> localstack.getEndpoint().toString());
    }

    /**
     * Get local url with port for integration tests
     */
    protected String getUrl(String path) {
        return "%s:%s%s".formatted(LOCALHOST, port, path);
    }
}