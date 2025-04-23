//date: 2025-04-23T17:02:37Z
//url: https://api.github.com/gists/6e995dc20598ae8c58361606da608781
//owner: https://api.github.com/users/kavicastelo

@RestController
@RequestMapping("/api")
public class GreetingController {
    @GetMapping("/hello")
    public Greeting sayHello() {
        return new Greeting("Hello, Intern!");
    }

    @GetMapping("/hello/{name}")
    public Greeting greetUser(@PathVariable String name) {
        return new Greeting("Hi, " + name + " 👋");
    }
}
