//date: 2022-09-16T23:01:23Z
//url: https://api.github.com/gists/c50fd577ad3633dcd0f437eb1534e3dd
//owner: https://api.github.com/users/danyllosoareszup

package com.example.exposicaolivrosclient.client;

import java.time.LocalDateTime;

public class DetalhaAutorResponse {

    private final Long id;
    private final String nome;
    private final String email;
    private final String descricao;
    private final LocalDateTime criadoEm;

    public DetalhaAutorResponse(Long id, String nome, String email, String descricao, LocalDateTime criadoEm) {
        this.id = id;
        this.nome = nome;
        this.email = email;
        this.descricao = descricao;
        this.criadoEm = criadoEm;
    }


    public Long getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public String getEmail() {
        return email;
    }

    public String getDescricao() {
        return descricao;
    }

    public LocalDateTime getCriadoEm() {
        return criadoEm;
    }
}