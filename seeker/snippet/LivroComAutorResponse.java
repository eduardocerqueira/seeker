//date: 2022-09-06T16:51:19Z
//url: https://api.github.com/gists/106f2f3445b5b71bf271a119a4f61425
//owner: https://api.github.com/users/joaosazup

public record LivroComAutorResponse(
        Long id, String nome, String descricao, String isbn, String publicadoEm, Author autor
) {
    LivroComAutorResponse(LivroResponse livroResponse, AutorResponse autorResponse) {
        this(livroResponse.id(), livroResponse.nome(), livroResponse.descricao(), livroResponse.isbn(),
                livroResponse.publicadoEm(), new Author(autorResponse.nome(), autorResponse.email())
        );
    }

    private record Author(String nome, String email) {
    }
}
