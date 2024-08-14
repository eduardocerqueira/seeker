//date: 2024-08-14T19:10:22Z
//url: https://api.github.com/gists/e7285579e3c95a4a8e40940a4ff727e4
//owner: https://api.github.com/users/pbrumblay

package com.tyrconsulting.identityparser;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.ExecutionException;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;

import com.google.cloud.documentai.v1.DocumentProcessorServiceClient;
import com.google.cloud.documentai.v1.DocumentProcessorServiceSettings;
import com.google.cloud.documentai.v1.Document;
import com.google.cloud.documentai.v1.ProcessRequest;
import com.google.cloud.documentai.v1.ProcessResponse;
import com.google.cloud.documentai.v1.RawDocument;
import com.google.protobuf.ByteString;
import com.google.cloud.documentai.v1.Document.Entity;
import java.nio.file.Files;
import java.util.concurrent.TimeoutException;

@SpringBootApplication
public class IdentityProofApplication {

    public static void main(String[] args) {
        SpringApplication.run(IdentityparserApplication.class, args);
    }

    @Bean
    public CommandLineRunner commandLineRunner(ApplicationContext ctx)
            throws IOException, InterruptedException, ExecutionException, TimeoutException {
        return args -> {
            String projectId = ""; //TODO: project id
            String location = "us"; // Format is "us" or "eu".
            String processorId = ""; //TODO: processor id format is something like 'aaaa55552222ffff'
            String filePath = ""; //TODO: file path
            quickStart(projectId, location, processorId, filePath);
        };
    }

    private void quickStart(String projectId, String location, String processorId, String filePath)
            throws IOException, InterruptedException, ExecutionException, TimeoutException {

        // Initialize client that will be used to send requests. This client only needs
        // to be created
        // once, and can be reused for multiple requests. After completing all of your
        // requests, call
        // the "close" method on the client to safely clean up any remaining background
        // resources.
        String endpoint = String.format("%s-documentai.googleapis.com:443", location);
        DocumentProcessorServiceSettings settings = DocumentProcessorServiceSettings.newBuilder().setEndpoint(endpoint)
                .build();
        try (DocumentProcessorServiceClient client = DocumentProcessorServiceClient.create(settings)) {
            // The full resource name of the processor, e.g.:
            // projects/project-id/locations/location/processor/processor-id
            // You must create new processors in the Cloud Console first
            String name = String.format("projects/%s/locations/%s/processors/%s", projectId, location, processorId);

            // Read the file.
            byte[] imageFileData = Files.readAllBytes(Paths.get(filePath));

            // Convert the image data to a Buffer and base64 encode it.
            ByteString content = ByteString.copyFrom(imageFileData);

            RawDocument document = RawDocument.newBuilder().setContent(content).setMimeType("image/png").build();

            // Configure the process request.
            ProcessRequest request = ProcessRequest.newBuilder().setName(name).setRawDocument(document).build();

            // Recognizes text entities in the PDF document
            ProcessResponse result = client.processDocument(request);
            Document documentResponse = result.getDocument();

            for (Entity entity : documentResponse.getEntitiesList()) {
                System.out.println(entity.getType());
                System.out.println("---: " + entity.getMentionText());
                switch (entity.getType()) {
                    case "fraud_signals_suspicious_words":
                    case "evidence_suspicious_word":
                        System.out.println(entity);
                        break;
                    case "fraud_signals_is_identity_document":
                    case "fraud_signals_image_manipulation":
                    case "fraud_signals_online_duplicate":
                    default:
                        break;
                }
            }
        }
    }
}