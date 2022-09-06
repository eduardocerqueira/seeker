//date: 2022-09-06T16:51:19Z
//url: https://api.github.com/gists/106f2f3445b5b71bf271a119a4f61425
//owner: https://api.github.com/users/joaosazup

@RestController
public class DetalharLivroController {
    private final LivrosClient livrosClient;
    private final AutoresClient autoresClient;

    public DetalharLivroController(LivrosClient livrosClient, AutoresClient autoresClient) {
        this.livrosClient = livrosClient;
        this.autoresClient = autoresClient;
    }

    @GetMapping("/api/livros/{livroId}")
    public ResponseEntity<?> listar(@PathVariable Long livroId) {
        LivroResponse livroResponse = livrosClient.getLivro(livroId);
        AutorResponse autorResponse = autoresClient.getAutor(livroResponse.autorId());
        LivroComAutorResponse livroComAutorResponse = new LivroComAutorResponse(livroResponse, autorResponse);
        return ResponseEntity.ok(livroComAutorResponse);
    }
}
