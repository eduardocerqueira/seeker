//date: 2023-04-21T16:53:51Z
//url: https://api.github.com/gists/dd6cb36ae2d252252801f20092a50c25
//owner: https://api.github.com/users/TejAllam

import java.util.*;

public class Threads {
    
    public static void workerThreads() {
        // Simulating thread execution
        try {
            Thread.sleep(1000);
        } catch (Exception e){}
    }

    public static void main(String[] args) throws Exception {
        int MAX_THREADS = 10;

        Thread thred = null;

        for (int i = 0;i< MAX_THREADS;i++) {
            thred = new Thread( Threads::workerThreads);
            thred.start();
        }

        System.out.println(" Starting thread " + thred);
        thred.join();
        System.out.println(" End thread " );
    }
}
