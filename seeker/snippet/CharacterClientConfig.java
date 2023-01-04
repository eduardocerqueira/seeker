//date: 2023-01-04T16:49:51Z
//url: https://api.github.com/gists/7aa56a6bc30a5790857e254848be94f1
//owner: https://api.github.com/users/bjerat

@Configuration
public class CharacterClientConfig {

    @Bean
    public CharacterClient characterClient(CharacterClientProperties properties) {
        WebClient webClient = WebClient.builder()
                .baseUrl(properties.getUrl())
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .defaultStatusHandler(
                        httpStatusCode -> HttpStatus.NOT_FOUND == httpStatusCode,
                        response -> Mono.empty())
                .defaultStatusHandler(
                        HttpStatusCode::is5xxServerError,
                        response -> Mono.error(new ExternalCommunicationException(response.statusCode().value())))
                .build();

        return HttpServiceProxyFactory
                .builder(WebClientAdapter.forClient(webClient))
                .build()
                .createClient(CharacterClient.class);
    }

}