//date: 2025-04-15T16:45:05Z
//url: https://api.github.com/gists/ea9b69f1a0f67570d1dbe278523f2c3d
//owner: https://api.github.com/users/Psor1107

// Mural.java
package br.ufscar.dc.dsw;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.logging.Logger;

public class Mural {
    private static Logger logger = Logger.getLogger(Mural.class.getName());
    private List<Mensagem> mensagens;

    // construtor
    public Mural() {
        logger.info("Mural Construtor");
        mensagens = Collections.synchronizedList(new ArrayList<>());

        // simula 2 mensagens existentes
        mensagens.add(new Mensagem(1, "andre", "turma", "Oi pesssoal!", new Date()));
        mensagens.add(new Mensagem(2, "Terminator", "John Connor", "I'll be back!", new Date()));
    }

    public List<Mensagem> getMensagens() {
        synchronized (mensagens) {
            return new ArrayList<>(mensagens);
        }
    }    

    public void addMensagem(String de, String para, String texto) {
        synchronized (mensagens) {
            int iid = mensagens.size() + 1;
            var novaMsg = new Mensagem(iid, de, para, texto, new Date());
            mensagens.add(novaMsg);
        }
    }    
}