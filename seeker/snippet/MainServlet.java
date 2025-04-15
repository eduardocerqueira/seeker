//date: 2025-04-15T16:45:05Z
//url: https://api.github.com/gists/ea9b69f1a0f67570d1dbe278523f2c3d
//owner: https://api.github.com/users/Psor1107

// MainServlet.java
package br.ufscar.dc.dsw;

import java.io.IOException;
import java.util.logging.Logger;

import io.vavr.control.Option;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebServlet(urlPatterns = {"/*"})
public class MainServlet extends HttpServlet {

    private Mural mural = new Mural();

    private static Logger logger = Logger.getLogger(MainServlet.class.getName());

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        processRequest(req, resp);
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        processRequest(req, resp);
    }

    public void processRequest(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String rota = Option.of(req.getPathInfo()).getOrElse("");
        
        Manager manager = new Manager(req, resp);

        if (rota.equals("/postar")) {

            logger.info("Rota /postar");
            
            manager.post(mural);

        } else if (rota.equals("/listar")) {
            
            logger.info("Rota /listar");

            manager.list(mural);
            
            logger.info("Mensagens atuais: " + mural.getMensagens());
        } else {
            
            logger.info("Rota não definida: " + rota);

            manager.error(rota);
        }
    }
}