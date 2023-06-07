//date: 2023-06-07T16:43:34Z
//url: https://api.github.com/gists/0187d1cbd5f2c132ae64951c58934338
//owner: https://api.github.com/users/Jafee

package worker.library.utils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpMethod;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MultiValueMap;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

@Slf4j
public class HttpClientUtils {

    private final HttpClient httpClient = HttpClient.newBuilder().build();
    private final ObjectMapper objectMapper = new ObjectMapper();

    public Optional<HttpResponse<String>> request(String url, HttpMethod method, Map<String, String> headers, MultiValueMap<String, String> uriParams, JsonNode body, int timeout) {
        URI uri;
        try {
            uri = new URI(url);
            if (!CollectionUtils.isEmpty(uriParams))
                uri = UriComponentsBuilder.fromUri(uri).queryParams(uriParams).build().toUri();
        } catch (URISyntaxException e) {
            log.error("uri build error", e);
            return Optional.empty();
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(uri)
                .timeout(Duration.of(timeout, ChronoUnit.SECONDS))
                .method(method.name(), Set.of(HttpMethod.POST, HttpMethod.PUT).contains(method)
                        ? HttpRequest.BodyPublishers.ofString(body.asText())
                        : HttpRequest.BodyPublishers.noBody())
                .headers(headers.entrySet().stream().map(v -> List.of(v.getKey(), v.getValue())).flatMap(List::stream).toArray(String[]::new))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (IOException | InterruptedException e) {
            log.error("http request error", e);
            return Optional.empty();
        }

        return Optional.of(response);
    }

}
