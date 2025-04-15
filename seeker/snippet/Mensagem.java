//date: 2025-04-15T16:45:05Z
//url: https://api.github.com/gists/ea9b69f1a0f67570d1dbe278523f2c3d
//owner: https://api.github.com/users/Psor1107

// Mensagem.java
package br.ufscar.dc.dsw;

import java.util.Date;

public record Mensagem(int iid,
                       String enviadoPor,
                       String enviadoPara,
                       String texto,
                       Date timestamp) { }