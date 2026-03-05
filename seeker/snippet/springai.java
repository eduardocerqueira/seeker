//date: 2026-03-05T18:31:56Z
//url: https://api.github.com/gists/d84e3447d34ffcd688930cd239248c67
//owner: https://api.github.com/users/sjmatta

@RestController
public class AgentController {

    private final ChatClient chatClient;
    private final WeatherTool weatherTool;

    public AgentController(ChatClient.Builder chatClientBuilder,
                           WeatherTool weatherTool) {
        this.weatherTool = weatherTool;
        this.chatClient = chatClientBuilder
                .defaultSystem("You are a helpful assistant. " +
                        "When users ask about weather, use the getWeather tool. " +
                        "Always provide a friendly, concise response.")
                .build();
    }

    @GetMapping("/chat")
    public String chat(
            @RequestParam(defaultValue = "What's the weather in Tokyo?") String message) {

        return chatClient.prompt()
                .user(message)
                .tools(weatherTool)   // make the tool available for this call
                .call()
                .content();
    }
}

@Service
public class WeatherTool {

    private static final Map<String, String> WEATHER_DATA = Map.of(
            "new york",    "62°F, partly cloudy",
            "london",      "55°F, overcast with light rain",
            "tokyo",       "72°F, sunny",
            "paris",       "59°F, mostly cloudy",
            "san francisco","58°F, foggy"
    );

    @Tool(description = "Get the current weather for a given city. " +
                         "Returns temperature and conditions.")
    public String getWeather(
            @ToolParam(description = "The city name, e.g. 'London' or 'New York'")
            String city) {

        String weather = WEATHER_DATA.get(city.toLowerCase().trim());

        if (weather != null) {
            return "Weather in %s: %s".formatted(city, weather);
        }
        return "Weather data not available for '%s'. Try: New York, London, Tokyo, Paris, or San Francisco.".formatted(city);
    }
}