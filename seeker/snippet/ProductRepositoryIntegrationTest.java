//date: 2025-05-27T16:56:45Z
//url: https://api.github.com/gists/3b94ebda1b5f88ec46b455dbce0da1ea
//owner: https://api.github.com/users/rdurelli

package com.br.ufla.sigrado.testcontainer;

import com.br.ufla.sigrado.testcontainer.models.Product;
import com.br.ufla.sigrado.testcontainer.repositories.ProductRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.support.TestPropertySourceUtils;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers // Habilita a integração do Testcontainers com JUnit 5 [cite: 17]
@DataJpaTest // Foca nos componentes JPA, desabilita auto-configuração completa
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE) // Desabilita o uso do H2 em memória
@ContextConfiguration(initializers = ProductRepositoryIntegrationTest.DataSourceInitializer.class) // Inicializador para propriedades dinâmicas
public class ProductRepositoryIntegrationTest {

    // Define o contêiner PostgreSQL.
    // 'static' faz com que o contêiner seja iniciado uma vez para todos os testes na classe [cite: 21]
    @Container
    public static PostgreSQLContainer<?> postgresContainer = new PostgreSQLContainer<>("postgres:13.2-alpine") // [cite: 17]
            .withDatabaseName("testdb") // [cite: 17]
            .withUsername("testuser") // [cite: 17]
            .withPassword("testpass"); // [cite: "**********"

    // Inicializador para fornecer as propriedades do datasource dinamicamente para o Spring
    // Isso é crucial porque as portas do contêiner são dinâmicas.
    public static class DataSourceInitializer implements ApplicationContextInitializer<ConfigurableApplicationContext> {
        @Override
        public void initialize(ConfigurableApplicationContext applicationContext) {
            TestPropertySourceUtils.addInlinedPropertiesToEnvironment(
                    applicationContext,
                    "spring.datasource.url=" + postgresContainer.getJdbcUrl(), // [cite: 23]
                    "spring.datasource.username=" + postgresContainer.getUsername(), // [cite: 23]
                    "spring.datasource.password=" + postgresContainer.getPassword(), // [cite: "**********"
                    "spring.jpa.hibernate.ddl-auto=create-drop"
            );
        }
    }

    @Autowired
    private ProductRepository productRepository;

    @Test
    void whenSaveProduct_thenProductIsPersisted() {
        // Given
        Product newProduct = new Product("Laptop Gamer", 7500.00);

        // When
        Product savedProduct = productRepository.save(newProduct);

        // Then
        assertThat(savedProduct).isNotNull();
        assertThat(savedProduct.getId()).isNotNull();
        assertThat(savedProduct.getName()).isEqualTo("Laptop Gamer");

        Product foundProduct = productRepository.findById(savedProduct.getId()).orElse(null);
        assertThat(foundProduct).isNotNull();
        assertThat(foundProduct.getName()).isEqualTo("Laptop Gamer");
    }

    @Test
    void whenFindAll_thenReturnProductList() {
        // Given
        productRepository.save(new Product("Mouse Sem Fio", 150.00));
        productRepository.save(new Product("Teclado Mecânico", 450.00));

        // When
        Iterable<Product> products = productRepository.findAll();

        // Then
        assertThat(products).hasSize(2);
    }
}
