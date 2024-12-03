//date: 2024-12-03T16:52:53Z
//url: https://api.github.com/gists/79eef5d8f305bef479edef1ccd09659b
//owner: https://api.github.com/users/acostaRossi

import java.net.*;
import java.util.Random;

public class UDPServer {
    public static void main(String[] args) {
        int port = 9876;
        Random random = new Random();
        int targetNumber = random.nextInt(100) + 1; // Numero casuale tra 1 e 100

        System.out.println("Server avviato. Numero da indovinare: " + targetNumber);

        try (DatagramSocket serverSocket = new DatagramSocket(port)) {
            byte[] receiveBuffer = new byte[1024];
            byte[] sendBuffer;

            while (true) {
                // Ricezione dei dati dal client
                DatagramPacket receivePacket = new DatagramPacket(receiveBuffer, receiveBuffer.length);
                serverSocket.receive(receivePacket);

                String clientMessage = new String(receivePacket.getData(), 0, receivePacket.getLength());
                System.out.println("Ricevuto dal client: " + clientMessage);

                // Analisi del messaggio del client
                int guessedNumber;
                try {
                    guessedNumber = Integer.parseInt(clientMessage.trim());
                } catch (NumberFormatException e) {
                    guessedNumber = -1; // Valore non valido
                }

                String response;
                if (guessedNumber == targetNumber) {
                    response = "Corretto! Hai indovinato!";
                    System.out.println("Numero indovinato dal client!");
                    targetNumber = random.nextInt(100) + 1; // Nuovo numero da indovinare
                    System.out.println("Nuovo numero da indovinare: " + targetNumber);
                } else if (guessedNumber < targetNumber) {
                    response = "Troppo basso!";
                } else {
                    response = "Troppo alto!";
                }

                // Invio della risposta al client
                InetAddress clientAddress = receivePacket.getAddress();
                int clientPort = receivePacket.getPort();
                sendBuffer = response.getBytes();

                DatagramPacket sendPacket = new DatagramPacket(sendBuffer, sendBuffer.length, clientAddress, clientPort);
                serverSocket.send(sendPacket);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
