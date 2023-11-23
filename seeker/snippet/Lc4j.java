//date: 2023-11-23T16:59:40Z
//url: https://api.github.com/gists/df7d547bf109029f8c122bfb313ad33b
//owner: https://api.github.com/users/fintanmm

///usr/bin/env jbang "$0" "$@" ; exit $?
// //DEPS <dependency1> <dependency2>

//DEPS info.picocli:picocli:4.5.0
//DEPS info.picocli:picocli-codegen:4.5.0
//DEPS ch.qos.reload4j:reload4j:1.2.19
//DEPS dev.langchain4j:langchain4j:0.24.0
//DEPS dev.langchain4j:langchain4j-embeddings:0.24.0
//DEPS dev.langchain4j:langchain4j-ollama:0.24.0

import ai.djl.util.Pair;
import dev.langchain4j.chain.ConversationalRetrievalChain;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.language.StreamingLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingLanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;
import picocli.CommandLine;
import picocli.CommandLine.Command;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

import static java.time.Duration.*;
import static java.util.concurrent.TimeUnit.SECONDS;

@Command(name = "lc4j", mixinStandardHelpOptions = true, version = "lc4j 0.0.1", description = "lc4j made with jbang")
public class Lc4j implements Runnable {

    @CommandLine.Option(names = {"-b", "--base-url"}, description = "Base url", defaultValue = "http://localhost:11434")
    private String baseUrl;

    @CommandLine.Option(names = {"-m", "--model"}, description = "Model name", defaultValue = "zephyr")
    private String modelName;

    @CommandLine.Option(names = {"-q", "--question"}, description = "Question", defaultValue = "What is the capital of Germany?")
    private String question;

    // set the temperature of the model
    @CommandLine.Option(names = {"-t", "--temperature"}, description = "Temperature", defaultValue = "0.5")
    private String temperature;

    @CommandLine.Option(names = {"-o", "--timeout"}, description = "Timeout", defaultValue = "30")
    private long timeout;
    
    // read the contents of a file
    @CommandLine.Option(names = {"-f", "--file"}, description = "File")
    private String file;

    private static final Logger logger = Logger.getLogger(Lc4j.class.getName());

    public static void main(String... args) {
        BasicConfigurator.configure();
        new CommandLine(new Lc4j()).execute(args);
    }

    @Override
    public void run() {
        Optional.ofNullable(question)
                .flatMap(q -> Optional.ofNullable(file)
                        .map(f -> new Pair<>(q, f)))
                .ifPresent(pair -> chatWithDocuments(pair.getKey(), pair.getValue()));
    }

    private void chatWithDocuments(String filePath, String value) {

        Document document = Document.from(filePath);
        EmbeddingModel embeddingModel = new OllamaEmbeddingModel(baseUrl, ofSeconds(timeout), modelName, 3);

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(500, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        ingestor.ingest(document);

        ConversationalRetrievalChain chain = ConversationalRetrievalChain.builder()
                .chatLanguageModel(OllamaChatModel.builder().modelId(modelName)
                        .temperature(Double.parseDouble(temperature))
                        .timeout(ofSeconds(timeout))
                        .build()).build();

        String answer = chain.execute(question);
        logger.info("Answer: " + answer);
    }

    private void askQuestion(String question) {
        CompletableFuture<String> futureAnswer = new CompletableFuture<>();
        CompletableFuture<Response<String>> futureResponse = new CompletableFuture<>();

        configure().generate(question, new StreamingResponseHandler<>() {

            private final StringBuilder answerBuilder = new StringBuilder();

            @Override
            public void onNext(String token) {
                logger.info("onNext: "**********"
                answerBuilder.append(token);
            }

            @Override
            public void onComplete(Response<String> response) {
                logger.info("onComplete: '" + response + "'");
                futureAnswer.complete(answerBuilder.toString());
                futureResponse.complete(response);
            }

            @Override
            public void onError(Throwable error) {
                futureAnswer.completeExceptionally(error);
                futureResponse.completeExceptionally(error);
            }
        });

        String answer = null;
        try {
            answer = futureAnswer.get(30, SECONDS);
        } catch (ExecutionException | TimeoutException | InterruptedException e) {
            logger.debug("Interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
        Response<String> response = null;
        try {
            response = futureResponse.get(30, SECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            logger.debug("Interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
        assert answer != null;
        assert !Objects.requireNonNull(response).content().isEmpty();
    }

    public StreamingLanguageModel configure() {
        return OllamaStreamingLanguageModel.builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .temperature(Double.parseDouble(temperature))
                .timeout(ofSeconds(timeout))
                .build();
    }
}
  }
}
