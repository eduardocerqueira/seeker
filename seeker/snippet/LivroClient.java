//date: 2022-09-16T23:01:23Z
//url: https://api.github.com/gists/c50fd577ad3633dcd0f437eb1534e3dd
//owner: https://api.github.com/users/danyllosoareszup

package com.example.exposicaolivrosclient.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

@FeignClient(
        name = "livrosClient",
        url = "${caminho.url}"
)
public interface LivroClient {

    @GetMapping("/api/livros/{id}")
    public DetalhaLivroResponse getLivro(@PathVariable Long id);

    @GetMapping("/api/autores/{id}")
    public DetalhaAutorResponse getAutor(@PathVariable Long id);
}