//date: 2024-02-21T16:57:51Z
//url: https://api.github.com/gists/c78369dca9761fd58fcd3c8b2b8d6674
//owner: https://api.github.com/users/delta-dev-software

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class MyController {

    @GetMapping("/hello")
    public String hello() {
        return "hello"; // Returns the view name "hello"
    }
}