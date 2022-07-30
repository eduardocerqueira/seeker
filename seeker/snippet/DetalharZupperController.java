//date: 2022-07-30T19:02:56Z
//url: https://api.github.com/gists/e6f8e70e67d29817a66b529870343e6b
//owner: https://api.github.com/users/danyllosoareszup

@RestController
public class DetalharZupperController {

    private final ZupperRepository zupperRepository;

    public DetalharZupperController(ZupperRepository zupperRepository) {
        this.zupperRepository = zupperRepository;
    }

    @GetMapping("/zupper/{id}")
    public ResponseEntity<?> detalharZupper(@PathVariable Long id) {

        Zupper zupper = zupperRepository.findById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        return ResponseEntity.ok(new ZupperResponse(zupper));
    }
}