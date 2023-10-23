//date: 2023-10-23T16:42:00Z
//url: https://api.github.com/gists/d039a96704262fe5d63b737a6c4f1264
//owner: https://api.github.com/users/Szustarol

//server code
public class Server {
    public static void main(String [] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(50004, 2000);
        int counter = 0;
        while(true){
            System.out.println(counter + " Starting to accept connection");
            Socket socket = serverSocket.accept();
            System.out.println(counter + " Connection accepted");
            socket.setTcpNoDelay(true);
            System.out.println(counter + " Closing connection");
            socket.close();
            counter++;
        }
    }
}
//client code
public class Client {
    public static void main(String [] args){
        int counter = 0;
        while(true){
            System.out.println(counter + " Connecting to server");
            try {
                Socket socket = new Socket("127.0.0.1", 50004);
                socket.setTcpNoDelay(true);
                socket.setSoTimeout(10000);
                System.out.println(counter + " Closing connection");
                socket.close();
                counter++;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
