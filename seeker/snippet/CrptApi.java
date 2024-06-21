//date: 2024-06-21T16:51:53Z
//url: https://api.github.com/gists/1c5797f3c166790cc0aa02d9dd6f78c3
//owner: https://api.github.com/users/EnterCapchaCode

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.vavr.control.Try;
import lombok.Getter;
import lombok.Setter;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

class CrptApi {
    private static volatile CrptApi instance;
    private final Semaphore semaphore;
    private final ScheduledExecutorService scheduler;
    private final WebClient webClient;
    private final ObjectMapper objectMapper;

    private static final String BASE_URL = "https://ismp.crpt.ru/api/v3/lk";
    private static final int SCHEDULER_DEFAULT_POOL_SIZE = 1;

    private CrptApi(TimeUnit timeUnit, int requestLimit) {
        this.semaphore = new Semaphore(requestLimit, true);
        this.scheduler = Executors.newScheduledThreadPool(SCHEDULER_DEFAULT_POOL_SIZE);
        this.objectMapper = new ObjectMapper();
        this.webClient = WebClient.builder()
                .baseUrl(BASE_URL)
                .build();

        setUpSemaphoreScheduler(timeUnit, requestLimit - semaphore.availablePermits());
    }

    public static CrptApi getInstance(TimeUnit timeUnit, int requestLimit) {
        if (instance == null) {
            synchronized (CrptApi.class) {
                if (instance == null) {
                    instance = new CrptApi(timeUnit, requestLimit);
                }
            }
        }
        return instance;
    }


    public synchronized void createDocument(Document document, String signature) {
        Try.run(semaphore::acquire)
                .andThenTry(() -> {
                    Mono<String> requestBody = Mono.just(objectMapper.writeValueAsString(document));
                    String responseBody = webClient.post()
                            .uri(uriBuilder -> uriBuilder.path("documents/create").build())
                            .contentType(MediaType.APPLICATION_JSON)
                            .header("Signature", signature)
                            .body(requestBody, String.class)
                            .retrieve()
                            .bodyToMono(String.class)
                            .block();

                    System.out.println(responseBody);
                })
                .onFailure(InterruptedException.class, e -> {
                    Thread.currentThread().interrupt();
                    System.err.println("Thread was interrupted: " + e.getMessage());
                })
                .onFailure(JsonProcessingException.class, e -> {
                    System.err.println("Failed to process JSON: " + e.getMessage());
                })
                .andFinally(semaphore::release);
    }

    public void shutdown() {
        scheduler.shutdown();
    }

    private void setUpSemaphoreScheduler(TimeUnit timeUnit, int permits) {
        long timeIntervalMillis = timeUnit.toMillis(1);
        this.scheduler.scheduleAtFixedRate(() -> semaphore.release(permits),
                timeIntervalMillis, timeIntervalMillis, TimeUnit.MILLISECONDS);
    }

    @Getter
    @Setter
    static class Document {
        private String description;
        private String doc_id;
        private String doc_status;
        private String doc_type;
        private boolean importRequest;
        private String owner_inn;
        private String participant_inn;
        private String producer_inn;
        private String production_date;
        private String production_type;
        private Product[] products;
        private String reg_date;
        private String reg_number;

        @Getter
        @Setter
        static class Product {
            private String certificate_document;
            private String certificate_document_date;
            private String certificate_document_number;
            private String owner_inn;
            private String producer_inn;
            private String production_date;
            private String tnved_code;
            private String uit_code;
            private String uitu_code;
        }
    }
}