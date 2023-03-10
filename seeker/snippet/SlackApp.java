//date: 2023-03-10T16:46:45Z
//url: https://api.github.com/gists/9186dac2eda267140d9e4ef163f2de2b
//owner: https://api.github.com/users/Davide453

package hello;

import com.slack.api.bolt.App;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SlackApp {
    @Bean
    public App initSlackApp() {
        App app = new App().asOAuthApp(true);
        app.command("/hello-oauth-app", (req, ctx) -> {
            return ctx.ack("What's up?");
        });
        return app;
    }
}