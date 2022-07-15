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
public class SCServer {
    
}
*/
import java.io.*; 
import java.net.*;
import java.sql.Timestamp;
import java.sql.Time;
import java.sql.*;


public class SCServer {

    public static void main(String args[]) throws Exception {
        //Variable declarations
        InetAddress lclhost;
        lclhost = InetAddress.getLocalHost();
        long maxtime, skewtime, datatime;
        String maxtimestr, skewtimestr;
        BufferedReader br;
        
        // Client server Instantiation / create an object for the client server      
        ClntServer ser = new ClntServer(lclhost);
        
        // Get user inputs for the sync clock
        System.out.println("Enter the maximum time");
        br = new BufferedReader(new InputStreamReader(System.in));
        maxtimestr = br.readLine();
        System.out.println("Enter the maximum skew time");
        br = new BufferedReader(new InputStreamReader(System.in));
        skewtimestr = br.readLine();
        maxtime = Long.parseLong(maxtimestr);
        skewtime = Long.parseLong(skewtimestr);
        
        // as long as the server is running...
        while (true) {
            // Get the current time
            datatime = System.currentTimeMillis();
            
            // Get G(Timestamp) by calculating the difference
            long G = datatime - maxtime - skewtime;
            // Print  timestamp
            System.out.println("G =" + G);
            // pass timestamp, recorded data & port to client
            ser.setTimeStamp(new Timestamp(G));
            ser.recPort(8001);
            ser.recData();
            
        }
    }
}



// create the client server class.
class ClntServer {
    
    // declarations
    InetAddress lclhost; // localhost
    int recport,sendPort; // receive &send port variables
    Timestamp obtmp; // object timestamp

    // Initialization of the client server class
    ClntServer(InetAddress lclhost) {
        this.lclhost = lclhost;
    }

    void recPort(int recport) {
        this.recport = recport;
    }
    void sendPort(int sendPort) {
        this.sendPort = sendPort;
    }

    void setTimeStamp(Timestamp obtmp) {
        this.obtmp = obtmp;
    }

    // This function records data from the client
    void recData() throws Exception {
        // Variabled declarations
        String msgstr = "";
        DatagramSocket ds;
        DatagramPacket dp;
        BufferedReader br;
        
        // Assignments
        byte buf[] = new byte[256];
        ds = new DatagramSocket(recport); // create a new data socket object
        dp = new DatagramPacket(buf, buf.length); // create a new data packet object
        ds.receive(dp);
        ds.close();
        msgstr = new String(dp.getData(), 0, dp.getLength());
        System.out.println(msgstr);
        
        // calculate the difference in timestamp and accept or reject
        Timestamp obtmp = new Timestamp(Long.parseLong(msgstr));
        if (this.obtmp.before(obtmp) == true) {
            System.out.println("The Message is accepted");
        } else {
            System.out.println("The Message is rejected");
        }
        
    }
}

