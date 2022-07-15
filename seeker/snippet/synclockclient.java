//date: 2022-07-15T17:07:20Z
//url: https://api.github.com/gists/401e28f84969b8bbaf1000792632eaf6
//owner: https://api.github.com/users/SebastianOpiyo

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package syncclock;

/**
 *
 * @author mshiyuka
 */
/*
public class SCClient {
    
}
*/

import java.io.*; 
import java.net.*;


public class SCClient {

    public static void main(String args[]) throws Exception {
        InetAddress lclhost;
        lclhost = InetAddress.getLocalHost();

        while (true) {
            Client cntl = new Client(lclhost);
            cntl.sendPort(9001);
            cntl.sendData();
        }
    }
}

class Client {

    InetAddress lclhost;
    int sendport,recport;
   
    // initialization functions
    Client(InetAddress lclhost) {
        this.lclhost = lclhost;
    }

    void sendPort(int sendport) {
        this.sendport = sendport;
    }

    // Send captured data function
    void sendData() throws Exception {
        DatagramPacket dp;
        DatagramSocket ds;
        BufferedReader br;
        br = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("Enter the data");
        String str = br.readLine();
        ds = new DatagramSocket(sendport);
       
       
        dp = new DatagramPacket(str.getBytes(), str.length(), lclhost, sendport-1000);
        ds.send(dp);
        ds.close();
    }
}
