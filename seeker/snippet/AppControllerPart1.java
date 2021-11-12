//date: 2021-11-12T17:00:05Z
//url: https://api.github.com/gists/533257283be97e819a7022b65a8091f3
//owner: https://api.github.com/users/baso53

@RestController
@RequestMapping("/app")
public class AppController {

    @GetMapping(path = "/test")
    public String test(Principal principal) {
        return principal.getName();
    }
}