//date: 2025-04-15T16:45:05Z
//url: https://api.github.com/gists/ea9b69f1a0f67570d1dbe278523f2c3d
//owner: https://api.github.com/users/Psor1107

//Manager.java
package br.ufscar.dc.dsw;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;

import io.vavr.control.Option;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public class Manager {

    private HttpServletRequest req;
    private HttpServletResponse resp;

    public Manager() {
    }

    public Manager(HttpServletRequest req, HttpServletResponse resp) {
        this.req = req;
        this.resp = resp;
    }

    public void post(Mural mural) throws IOException, ServletException {
        var enviadoPor = Option.of(req.getParameter("enviadoPor")).getOrElse("");
        if (enviadoPor.equals("")) {
            enviadoPor = "Desconhecido";
        }
        var enviadoPara = Option.of(req.getParameter("enviadoPara")).getOrElse("");
        if (enviadoPara.equals("")) {
            enviadoPara = "Desconhecido";
        }
        var texto = Option.of(req.getParameter("texto")).getOrElse("");
        if (texto.equals("")) {
            texto = "Sem mensagem escrita.";
        }

        mural.addMensagem(enviadoPor, enviadoPara, texto);
        // redireciona para a rota listar
        resp.sendRedirect("listar");
    }

    public void list(Mural mural) throws IOException, ServletException {
        String htmlTemplate = Files.readString(Path.of("C:/Users/Psor1/Desktop/dsw1/httprequest/pages/mural.html"));
        StringBuilder mensagensHtml = new StringBuilder();
        mural.getMensagens().forEach(mensagem -> {
            mensagensHtml.append(String.format(
                "<div><div><strong>De:</strong> %s &nbsp; <strong>Para:</strong> %s (em %s)</div>",
                mensagem.enviadoPor(),
                mensagem.enviadoPara(),
                mensagem.timestamp()
            ));
            mensagensHtml.append(String.format("<div>%s</div></div><br>", mensagem.texto()));
        });
        String htmlFinal = htmlTemplate.replace("__MENSAGENS__", mensagensHtml.toString());
        resp.setContentType("text/html;charset=UTF-8");
        PrintWriter out = resp.getWriter();
        out.println(htmlFinal);
        out.close();
    }

    public void error(String rota) throws IOException, ServletException {
        String htmlContent = Files.readString(Path.of("C:/Users/Psor1/Desktop/dsw1/httprequest/pages/error.html"));
        htmlContent = htmlContent.replace("__ROTA__", rota);
        resp.setContentType("text/html;charset=UTF-8");
        PrintWriter out = resp.getWriter();
        out.println(htmlContent);
        out.close();
    }
}
