//date: 2024-06-24T16:54:33Z
//url: https://api.github.com/gists/dd97fda73250b9f9f8fa71a2f966dc05
//owner: https://api.github.com/users/ilvi89

package com.test;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Data;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;
public class CrptApi {
    private final HttpClient httpClient;
    private final Semaphore semaphore;
    private final ObjectMapper objectMapper;

    public CrptApi(Duration duration, int requestLimit) {
        this.httpClient = HttpClient.newHttpClient();
        this.semaphore = new Semaphore(requestLimit);
        this.objectMapper = new ObjectMapper();

        long interval = duration.toMillis();

        Thread permitReleaser = new Thread(() -> {
            while (true) {
                try {
                    TimeUnit.MILLISECONDS.sleep(interval);
                    semaphore.release(requestLimit - semaphore.availablePermits());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        permitReleaser.setDaemon(true);
        permitReleaser.start();
    }

    public String createDocument(DocumentDTO document) throws Exception {
        semaphore.acquire();

        String jsonDocument = objectMapper.writeValueAsString(document);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI("https://ismp.crpt.ru/api/v3/lk/documents/create"))
                .header("Content-Type", "application/json")
                .header("Signature", "signature")
                .POST(HttpRequest.BodyPublishers.ofString(jsonDocument))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        return response.body();
    }

    @Data
    public static class DocumentDTO {

        @JsonProperty("description")
        private Description description;

        @JsonProperty("doc_id")
        private String docId;

        @JsonProperty("doc_status")
        private String docStatus;

        @JsonProperty("doc_type")
        private String docType;

        @JsonProperty("importRequest")
        private boolean importRequest;

        @JsonProperty("owner_inn")
        private String ownerInn;

        @JsonProperty("participant_inn")
        private String participantInn;

        @JsonProperty("producer_inn")
        private String producerInn;

        @JsonProperty("production_date")
        private String productionDate;

        @JsonProperty("production_type")
        private String productionType;

        @JsonProperty("products")
        private List<Product> products;

        @JsonProperty("reg_date")
        private String regDate;

        @JsonProperty("reg_number")
        private String regNumber;

        @Data
        public static class Description {
            @JsonProperty("participantInn")
            private String participantInn;
        }

        @Data
        public static class Product {
            @JsonProperty("certificate_document")
            private String certificateDocument;

            @JsonProperty("certificate_document_date")
            private String certificateDocumentDate;

            @JsonProperty("certificate_document_number")
            private String certificateDocumentNumber;

            @JsonProperty("owner_inn")
            private String ownerInn;

            @JsonProperty("producer_inn")
            private String producerInn;

            @JsonProperty("production_date")
            private String productionDate;

            @JsonProperty("tnved_code")
            private String tnvedCode;

            @JsonProperty("uit_code")
            private String uitCode;

            @JsonProperty("uitu_code")
            private String uituCode;
        }
    }
}
