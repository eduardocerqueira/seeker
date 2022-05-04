//date: 2022-05-04T17:04:14Z
//url: https://api.github.com/gists/d07a45f9f2ba362dbcb718212f63122d
//owner: https://api.github.com/users/MouamleH

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.atomic.AtomicInteger;

public class ProxyServer implements Runnable {

    private final int port;
    private final AtomicInteger targetPort = new AtomicInteger();

    public ProxyServer(int port, int targetPort) {
        this.port = port;
        this.targetPort.set(targetPort);
    }

    public void setTargetPort(int targetPort) {
        this.targetPort.set(targetPort);
    }

    @Override
    public void run() {
        try {
            ServerSocket serverSocket = new ServerSocket(port);
            while (!Thread.interrupted()) {
                final Socket client = serverSocket.accept();
                new Thread(() -> {
                    try {
                        final InputStream clientInputStream = client.getInputStream();
                        final OutputStream clientOutputStream = client.getOutputStream();

                        try (Socket server = new Socket("localhost", targetPort.get())) {
                            final InputStream serverInputStream = server.getInputStream();
                            final OutputStream serverOutputStream = server.getOutputStream();

                            new Thread(() -> {
                                try {
                                    int clientBytesRead;
                                    byte[] request = new byte[1024 * 10];
                                    while ((clientBytesRead = clientInputStream.read(request)) != -1) {
                                        serverOutputStream.write(request, 0, clientBytesRead);
                                        serverOutputStream.flush();
                                    }
                                    serverOutputStream.close();
                                } catch (IOException ignored) {}
                            }).start();

                            int serverBytesRead;
                            byte[] response = new byte[1024 * 10];
                            while ((serverBytesRead = serverInputStream.read(response)) != -1) {
                                clientOutputStream.write(response, 0, serverBytesRead);
                                clientOutputStream.flush();
                            }
                            clientOutputStream.close();
                        } catch (IOException ignored) {}
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }).start();
            }
            serverSocket.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}