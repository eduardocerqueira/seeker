//date: 2022-11-24T17:02:30Z
//url: https://api.github.com/gists/ba6867b242bfe69835edb3e4eb8b7837
//owner: https://api.github.com/users/zeddo123

class UDPClient {
  public static void main(String[] args) {
    int n = 5;
    int port = 4000;
    byte[] buffer = String.valueof(n).getBytes();
    try{
      InetAddress address = InetAddress.getByName("add");
      DatagramSocket socket = new DatagramSocket();
      DatagramPacket packet = new DatagramPacket(buffer, buffer.length, address, port);
      socket.send(packet);
      
      buffer = new byte[UDP.MAX_SIZE];
      final DatagramPacket resp = new DatagramPacket(buffer, buffer.length);
      socket.receive(resp);
      int resp_n = Integer.parseInt(new String(resp.getData(), 0, resp.getLength()));
      socket.close();
    } catch (SocketException e) {
      throw new RuntimeException(e);
    } catch (UnknownHostException e) {
      throw new RuntimeException(e)
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
 }
} 