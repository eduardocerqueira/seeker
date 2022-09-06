//date: 2022-09-06T17:06:01Z
//url: https://api.github.com/gists/f104833acc19798912e8c1eab09f5e8b
//owner: https://api.github.com/users/thiagonuneszup

RestController
public class LivroEAutorDetalhadoController {

    @Autowired
    private LivrariaClient client;

    @GetMapping("/api/livros/{id}")
    public ResponseEntity<?> detalha(@PathVariable Long id){

        LivroResponseApiExterna livroResponse = client.detalhaLivro(id)
                .orElseThrow(()-> new ResponseStatusException(HttpStatus.NOT_FOUND,"livro nao encontrado"));

        AutorResponse autorResponse = client.detalhaAutor(livroResponse.getAutorId())
                .orElseThrow(()-> new ResponseStatusException(HttpStatus.NOT_FOUND,"autor nao encontrado"));


        return ResponseEntity.ok(new LivroResponse(livroResponse,autorResponse));

    }


}
