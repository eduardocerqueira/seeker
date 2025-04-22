//date: 2025-04-22T16:58:33Z
//url: https://api.github.com/gists/4d4300041d82ab7d306ccba7d789bb29
//owner: https://api.github.com/users/ArcureDev

@RestController
@RequestMapping("/api/ping")
public class PingController {

    @GetMapping
    public String ping() {
        return "ni dieu ni maître";
    }

}