//date: 2024-12-05T17:02:42Z
//url: https://api.github.com/gists/0cd5acc824db730c068a18840d18d8ba
//owner: https://api.github.com/users/acostaRossi

import java.io.*;
import java.net.*;
import java.util.*;

public class IRCServer {
    private static final int PORT = 6667; // Porta standard IRC
    private static Set<ClientHandler> clients = new HashSet<>();

    public static void main(String[] args) {
        System.out.println("IRC Server avviato sulla porta " + PORT);

        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("Nuovo client connesso: " + clientSocket.getInetAddress());
                ClientHandler clientHandler = new ClientHandler(clientSocket);
                clients.add(clientHandler);
                new Thread(clientHandler).start();
            }
        } catch (IOException e) {
            System.err.println("Errore nel server: " + e.getMessage());
        }
    }

    // Invia un messaggio a tutti i client connessi
    public static void broadcast(String message, ClientHandler sender) {
        for (ClientHandler client : clients) {
            if (client != sender) {
                client.sendMessage(message);
            }
        }
    }

    // Rimuove un client dalla lista quando si disconnette
    public static void removeClient(ClientHandler clientHandler) {
        clients.remove(clientHandler);
        System.out.println("Client disconnesso.");
    }

    // Classe per gestire un singolo client
    private static class ClientHandler implements Runnable {
        private Socket socket;
        private PrintWriter out;
        private String nickname;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out = new PrintWriter(socket.getOutputStream(), true);

                // Richiede il nickname
                out.println("Benvenuto nel server IRC! Inserisci il tuo nickname:");
                nickname = in.readLine();
                System.out.println(nickname + " si è unito alla chat.");
                broadcast(nickname + " è entrato nella chat!", this);

                // Legge i messaggi dal client
                String message;
                while ((message = in.readLine()) != null) {
                    if (message.equalsIgnoreCase("/quit")) {
                        break;
                    }
                    System.out.println(nickname + ": " + message);
                    broadcast(nickname + ": " + message, this);
                }
            } catch (IOException e) {
                System.err.println("Errore con il client: " + e.getMessage());
            } finally {
                try {
                    socket.close();
                } catch (IOException e) {
                    System.err.println("Errore durante la chiusura della connessione: " + e.getMessage());
                }
                removeClient(this);
                broadcast(nickname + " ha lasciato la chat.", this);
            }
        }

        // Invia un messaggio a questo client
        public void sendMessage(String message) {
            out.println(message);
        }
    }
}
