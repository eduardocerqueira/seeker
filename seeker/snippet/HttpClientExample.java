//date: 2021-11-26T16:56:07Z
//url: https://api.github.com/gists/102220a5ff8862a0d968174044253b91
//owner: https://api.github.com/users/minhdua

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class HttpClientExample {
    public static void main(String[] args) throws IOException, InterruptedException {
        //supports HTTP2
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(20))
                .build();
        HttpRequest getRequest = HttpRequest.newBuilder()
                .GET()
                .uri(URI.create("http://localhost:8080"))
                .build();
        HttpResponse<String> response = httpClient.send(getRequest,
                                                        HttpResponse.BodyHandlers.ofString());
        System.out.println(response.statusCode());
        System.out.println(response.body());
    }
}
