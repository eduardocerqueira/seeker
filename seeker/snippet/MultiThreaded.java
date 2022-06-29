//date: 2022-06-29T17:22:36Z
//url: https://api.github.com/gists/0112099a7bc76ca803a3af29c3f477a7
//owner: https://api.github.com/users/etemesi254

package org.example;


// Java code for thread creation by extending
// the Thread class
class MultithreadingDemo extends Thread {

    int num = 433452;
    public void run()
    {
        try {
            num /= Thread.currentThread().getId();
  
            System.out.println(num);
        }
        catch (Exception e) {
            // Throwing an exception
            System.out.println("Exception is caught");
        }
    }
}

// Main Class
public class MultiThreaded {
    public static void main(String[] args)
    {
        int n = 8; // Number of threads
        for (int i = 0; i < n; i++) {
            MultithreadingDemo object
                    = new MultithreadingDemo();
            object.start();
        }
    }
}
