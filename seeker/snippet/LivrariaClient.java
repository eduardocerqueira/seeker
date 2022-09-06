//date: 2022-09-06T17:04:54Z
//url: https://api.github.com/gists/e5cde6f2ffcd65ee4e31ac7c723a2041
//owner: https://api.github.com/users/thiagonuneszup

@FeignClient(
        name = "livrariaClient",
        url = "http://localhost:8080/oauth2-resourceserver-livraria/"
)

public interface LivrariaClient {

    @GetMapping("api/livros/{id}")
    public Optional<LivroResponseApiExterna> detalhaLivro(@PathVariable Long id);

    @GetMapping("/api/autores/{id}")
    public Optional<AutorResponse> detalhaAutor(@PathVariable Long id);


}
