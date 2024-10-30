//date: 2024-10-30T17:10:48Z
//url: https://api.github.com/gists/df0279a2a615e101954dd455c6a7cccd
//owner: https://api.github.com/users/nkchauhan003

package com.cb.client;

import com.cb.Application;
import com.cb.client.response.InventoryResponse;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.mockito.MockitoAnnotations;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceTransactionManagerAutoConfiguration;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.web.client.RestTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Test class for InventoryClient.
 */
@SpringBootTest(classes = Application.class)
@EnableAutoConfiguration(exclude = {DataSourceAutoConfiguration.class,
        DataSourceTransactionManagerAutoConfiguration.class,
        HibernateJpaAutoConfiguration.class})
@Execution(ExecutionMode.SAME_THREAD)
class InventoryClientTest {

    @MockBean
    private RestTemplate restTemplate;

    @Value("${inventory.service.url}")
    private String inventoryServiceUrl;

    @MockBean
    private CircuitBreaker circuitBreaker;

    @Autowired
    private InventoryClient inventoryClient;

    @Autowired
    private CircuitBreakerRegistry circuitBreakerRegistry;

    /**
     * Sets up the test environment before each test.
     */
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    /**
     * Tests that getInventoryItem returns the item when the circuit breaker is closed.
     */
    @Test
    void getInventoryItem_ReturnsItem_WhenCircuitBreakerClosed() {
        circuitBreakerRegistry.circuitBreaker("inventory-circuit-breaker")
                .transitionToClosedState();
        when(restTemplate.getForObject("http://localhost:8081/inventory/1", InventoryResponse.class))
                .thenReturn(new InventoryResponse("1", "Item 1", 10));
        InventoryResponse result = inventoryClient.getInventoryItem("1");
        assertEquals("1", result.id());
        verify(restTemplate, times(1)).getForObject(anyString(), eq(InventoryResponse.class));
    }

    /**
     * Tests that getInventoryItem returns the default item when the circuit breaker is open.
     */
    @Test
    void getInventoryItem_ReturnsDefault_WhenCircuitBreakerOpen() {
        circuitBreakerRegistry.circuitBreaker("inventory-circuit-breaker")
                .transitionToOpenState();
        doThrow(new RuntimeException()).when(restTemplate).getForObject("http://localhost:8081/inventory/1", InventoryResponse.class);
        try {
            InventoryResponse result = inventoryClient.getInventoryItem("1");
            assertEquals("default", result.id());
        } catch (RuntimeException e) {
            assertEquals(CallNotPermittedException.class, e.getClass());
            verifyNoInteractions(restTemplate);
        }
    }

    /**
     * Tests that getInventoryItem returns the item when the circuit breaker is half-open.
     */
    @Test
    void getInventoryItem_ReturnsItem_WhenCircuitBreakerHalfOpen() {
        circuitBreakerRegistry.circuitBreaker("inventory-circuit-breaker")
                .transitionToOpenState();
        circuitBreakerRegistry.circuitBreaker("inventory-circuit-breaker")
                .transitionToHalfOpenState();
        doThrow(new RuntimeException()).when(restTemplate).getForObject("http://localhost:8081/inventory/1", InventoryResponse.class);
        try {
            InventoryResponse result = inventoryClient.getInventoryItem("1");
            assertEquals("default", result.id());
        } catch (RuntimeException e) {
            assertEquals(CallNotPermittedException.class, e.getClass());
            verifyNoInteractions(restTemplate);
        }
    }
}