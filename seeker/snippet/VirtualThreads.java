//date: 2025-06-26T17:14:20Z
//url: https://api.github.com/gists/4c1b8d1bea7c6c5f55d79cd808ddf79f
//owner: https://api.github.com/users/suneel-tokuri

import java.security.SecureRandom;
import java.util.concurrent.atomic.*;

public class VirtualThreads {

  private SecureRandom random1;
  private AtomicInteger count;

  public VirtualThreads() throws Exception {
      random1 = SecureRandom.getInstance("NativePRNG");

      count = new AtomicInteger();
  }

  public void run() throws Exception {

    int no_of_threads = 10;

    for (int i = 0; i < no_of_threads; i++) {

      Thread vthread = Thread.ofVirtual().

        name("vthread_" + i).

        start(new Runnable() {

          public void run() {
            try {
              String tname = Thread.currentThread().getName();

              int wt = random1.nextInt(8000);
              System.out.println("Thread id: <" + tname + "> " + wt + " ms");
              Thread.sleep(wt);
    
              int finishPos = count.incrementAndGet();
              System.out.println(tname + " finished @" + finishPos);

              if (finishPos == no_of_threads - 1) {
                synchronized(count) {
                  count.notify();
                }
              }
            } catch(InterruptedException e) {
              e.printStackTrace();
            }
          }
      });

    }

    
    synchronized(count) { 
      count.wait();
    }
  }


  public static void main(String[] args) {

    try {

      new VirtualThreads().run();

    } catch(Exception e) {
      e.printStackTrace();
    }
  }

}

/**
 * Sample output:
$ java VirtualThreads     
Thread id: <vthread_4> 1986 ms
Thread id: <vthread_5> 5098 ms
Thread id: <vthread_7> 7487 ms
Thread id: <vthread_2> 6822 ms
Thread id: <vthread_6> 5237 ms
Thread id: <vthread_3> 5921 ms
Thread id: <vthread_0> 6023 ms
Thread id: <vthread_1> 4292 ms
Thread id: <vthread_9> 6927 ms
Thread id: <vthread_8> 4044 ms
vthread_4 finished @1
vthread_8 finished @2
vthread_1 finished @3
vthread_5 finished @4
vthread_6 finished @5
vthread_3 finished @6
vthread_0 finished @7
vthread_2 finished @8
vthread_9 finished @9
*/