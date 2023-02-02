//date: 2023-02-02T16:52:04Z
//url: https://api.github.com/gists/22ef2fae47be4185217c8624ec0e6c60
//owner: https://api.github.com/users/erdalguzel

import java.io.FileNotFoundException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class Main {
	public static void main(String[] args) {

		HttpClient client = HttpClient.newBuilder().build();

		try {
			HttpRequest httpRequest = HttpRequest.newBuilder()
					.uri(new URI("https://www.google.com/"))
					.timeout(Duration.ofMinutes(2))
					.version(HttpClient.Version.HTTP_2)
					.header("Content-Type", "application/json")
					.build();

			CompletableFuture<HttpResponse<String>> response = client.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString());
			System.out.println(response.get().statusCode());
			System.out.println(response.get().body());
		} catch (URISyntaxException | InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}

	}
}