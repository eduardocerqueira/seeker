//date: 2022-09-16T23:01:23Z
//url: https://api.github.com/gists/c50fd577ad3633dcd0f437eb1534e3dd
//owner: https://api.github.com/users/danyllosoareszup

package com.example.exposicaolivrosclient.client;

import java.time.LocalDate;

public class DetalhaLivroResponse {

    private Long id;
    private String nome;
    private String descricao;
    private String isbn;
    private Long autorId;
    private LocalDate publicadoEm;


    public DetalhaLivroResponse(Long id, String nome, String descricao, String isbn, Long autorId, LocalDate publicadoEm) {
        this.id = id;
        this.nome = nome;
        this.descricao = descricao;
        this.isbn = isbn;
        this.autorId = autorId;
        this.publicadoEm = publicadoEm;
    }

    public DetalhaLivroResponse() {
    }

    public Long getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public String getDescricao() {
        return descricao;
    }

    public String getIsbn() {
        return isbn;
    }

    public Long getAutorId() {
        return autorId;
    }

    public LocalDate getPublicadoEm() {
        return publicadoEm;
    }
}
