//date: 2024-01-01T16:40:08Z
//url: https://api.github.com/gists/568ea680bcd9817ac33d2ee8ea0f426e
//owner: https://api.github.com/users/efekurbann

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

@Data
public class License {

    // replace this with your API-KEY
    private static final String API_KEY = "MY_API_KEY";
    
    // replace this with your IP.
    private static final String BASE_URL = "http://127.0.0.1:8080";
    
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL).connectTimeout(Duration.ofSeconds(5)).build();

    private final JavaPlugin plugin;
    private final String key;
    private final String product;

    public boolean check(boolean print) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(String.format(BASE_URL + "/public/license/enable?key=%s&product=%s", key, product)))
                    .header("Content-Type", "application/json")
                    .header("X-API-KEY", API_KEY)
                    .POST(HttpRequest.BodyPublishers.noBody())
                    .build();

            HttpResponse<String> response = CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
            String body = response.body();
            if (response.statusCode() == 200) {
                if (print)
                    plugin.getLogger().info("License verified, plugin enabled!");

                return true;
            } else if (response.statusCode() == 404 || response.statusCode() == 406) {
                SuccessResponse successResponse = GSON.fromJson(body, SuccessResponse.class);

                if (print) {
                    plugin.getLogger().info("Disabling the plugin.");
                    plugin.getLogger().info("Reason: " + successResponse.message);
                }
                plugin.getServer().getPluginManager().disablePlugin(plugin);
                return false;
            } else {
                if (print) {
                    plugin.getLogger().info("Disabling the plugin.");
                    plugin.getLogger().info("Reason: Could not reach the license servers!");
                }

                plugin.getServer().getPluginManager().disablePlugin(plugin);
                return false;
            }
        } catch (IOException | InterruptedException | URISyntaxException e) {
            if (print) {
                plugin.getLogger().info("Disabling the plugin.");
                plugin.getLogger().info("Reason: Could not reach the license servers!");
            }

            plugin.getServer().getPluginManager().disablePlugin(plugin);
            return false;
        }
    }

    @Data
    @AllArgsConstructor
    public static class SuccessResponse {

        private boolean success;
        private String message;

    }

}
