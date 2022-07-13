//date: 2022-07-13T17:14:34Z
//url: https://api.github.com/gists/b066866a0272c4c84c82ed735998cde2
//owner: https://api.github.com/users/sfariaNG

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

@EnableWebFluxSecurity
public class SecurityConfiguration {
    private static final String LOGOUT_URL = "/logout";
    private static final String LOGGED_OUT_PARAM = "success";

    private static final String LOGOUT_HTML = "/logout.html";
    private static final String LOGGED_OUT_HTML = "/loggedout.html";
    private static final String LOGOUT_CSS = "/logout.css";

    // Must be on the classpath, e.g. resources/static/logout.html
    @Value("classpath:/static" + LOGOUT_HTML)
    private Resource logout;
    @Value("classpath:/static" + LOGGED_OUT_HTML)
    private Resource loggedout;

    @Bean
    public RouterFunction<ServerResponse> logoutRoutes() {
        // Return custom logout pages and stylesheet
        return RouterFunctions
            .route(RequestPredicates.GET(LOGOUT_URL)
                .and(RequestPredicates.queryParam(LOGGED_OUT_PARAM, "").negate())
                .and(RequestPredicates.pathExtension("css").negate()),
                    request -> ServerResponse
                        .ok().contentType(MediaType.TEXT_HTML).bodyValue(logout)
            )
            .andRoute(RequestPredicates.GET(LOGOUT_URL)
                .and(RequestPredicates.queryParam(LOGGED_OUT_PARAM, "")),
                    request -> ServerResponse
                        .ok().contentType(MediaType.TEXT_HTML).bodyValue(loggedout)
            );
    }
}