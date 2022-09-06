//date: 2022-09-06T17:05:28Z
//url: https://api.github.com/gists/e190621d1bdf5be17f1a318a971cb94c
//owner: https://api.github.com/users/thiagonuneszup

public class OAuth2FeignRequestInterceptor implements RequestInterceptor {

    private static final Authentication ANONYMOUS_AUTHENTICATION = "**********"
            "anonymous", "anonymousUser", AuthorityUtils.createAuthorityList("ROLE_ANONYMOUS"));

    private final OAuth2AuthorizedClientManager authorizedClientManager;
    private final String clientRegistrationId;

    public OAuth2FeignRequestInterceptor(OAuth2AuthorizedClientManager authorizedClientManager, String clientRegistrationId) {
        this.authorizedClientManager = authorizedClientManager;
        this.clientRegistrationId = clientRegistrationId;
    }

    @Override
    public void apply(RequestTemplate request) {
        if (this.authorizedClientManager == null) {
            return;
        }

        OAuth2AuthorizeRequest authorizeRequest = OAuth2AuthorizeRequest
                .withClientRegistrationId(this.clientRegistrationId)
                .principal(ANONYMOUS_AUTHENTICATION)
                .build();

        OAuth2AuthorizedClient authorizedClient = this.authorizedClientManager.authorize(authorizeRequest);

        if (authorizedClient == null) {
            throw new IllegalStateException(
                    "This client uses an authorization grant type that's not supported by the " +
                            "configured provider. ClientRegistrationId = " + this.clientRegistrationId);
        }

        OAuth2AccessToken accessToken = "**********"
        request
                .header(HttpHeaders.AUTHORIZATION,
                        String.format("Bearer %s", accessToken.getTokenValue()));
    }
}
s", accessToken.getTokenValue()));
    }
}
