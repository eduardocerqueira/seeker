//date: 2023-03-10T16:46:45Z
//url: https://api.github.com/gists/9186dac2eda267140d9e4ef163f2de2b
//owner: https://api.github.com/users/Davide453

package hello;

import com.slack.api.bolt.App;
import com.slack.api.bolt.servlet.SlackOAuthAppServlet;

import javax.servlet.annotation.WebServlet;

@WebServlet("/slack/oauth/callback")
public class SlackOAuthCallbackController extends SlackOAuthAppServlet {
    public SlackOAuthCallbackController(App app) {
        super(app);
    }
}