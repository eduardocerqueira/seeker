//date: 2024-12-05T17:02:42Z
//url: https://api.github.com/gists/0cd5acc824db730c068a18840d18d8ba
//owner: https://api.github.com/users/acostaRossi

import java.io.*;
import java.net.*;

public class IRCClient {
    private static final String SERVER_ADDRESS = "localhost";
    private static final int SERVER_PORT = 6667;

    public static void main(String[] args) {
        try {
             Socket socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader console = new BufferedReader(new InputStreamReader(System.in));

            System.out.println("Connesso al server IRC!");
            System.out.println("Per uscire digita '/quit'");

            // Thread per leggere i messaggi dal server
            Thread serverListener = new Thread(() -> {
                try {
                    String message;
                    while ((message = in.readLine()) != null) {
                        System.out.println(message);
                    }
                } catch (IOException e) {
                    System.err.println("Connessione persa con il server.");
                }
            });
            serverListener.start();

            // Legge i messaggi dalla console e li invia al server
            String userInput;
            while ((userInput = console.readLine()) != null) {
                out.println(userInput);
                if (userInput.equalsIgnoreCase("/quit")) {
                    break;
                }
            }
        } catch (IOException e) {
            System.err.println("Errore nella connessione al server: " + e.getMessage());
        }
    }
}
