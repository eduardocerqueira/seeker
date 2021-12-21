//date: 2021-12-21T17:11:15Z
//url: https://api.github.com/gists/7f4a92d9b0514c59994b365289983103
//owner: https://api.github.com/users/renatoathaydes

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;

class Scratch {
    public static void main(String[] args) throws Exception {
        var singleThreadedExecutor = Executors.newSingleThreadExecutor();

        var startTime = System.currentTimeMillis();

        /*
         * We are trying to run several tasks using "async" methods... that ideally won't block as
         * they will run on another "asynchronous context".
         *
         * However, as you'll find out when running this example, that's a meaningless promise by Java
         * because as we only have one Thread to run the async tasks, every task will still "block"
         * the next... even when you have many Threads, it is not guaranteed by Java that this won't
         * happen. This makes the "async" methods almost meaningless unless the only thing you care
         * is not blocking until you explicitly call a blocking method (as I do here calling `get()` at the end).
         * But that's nearly never going to be the case, what matters is that a task doesn't block the other,
         * which can only be done by running them not async, but in completely independent Threads!
         */

        System.out.println("Running async 1");
        var future1 = CompletableFuture.runAsync(() -> {
            delay(1000);
            System.out.println("Future 1 done at " + (System.currentTimeMillis() - startTime));
        }, singleThreadedExecutor);
        
        System.out.println("Running async 2");
        var future2 = CompletableFuture.runAsync(() -> {
            delay(1000);
            // as both future 1 and future 2 were scheduled "at the same time", and we called the
            // non-blocking async method, both should end after around 1 second, right?
            System.out.println("Future 2 done at " + (System.currentTimeMillis() - startTime));
        }, singleThreadedExecutor);

        var future3 = future1.whenCompleteAsync((ok, err) -> {
            if (err != null) {
                err.printStackTrace();
            } else {
                System.out.println("completed future 1 at " + (System.currentTimeMillis() - startTime));
            }
        }, singleThreadedExecutor);
        
        var future4 = future2.whenCompleteAsync((ok, err) -> {
            if (err != null) {
                err.printStackTrace();
            } else {
                System.out.println("completed future 2 at " + (System.currentTimeMillis() - startTime));
            }
        }, singleThreadedExecutor);

        try {
            CompletableFuture.allOf(future3, future4).get();
        } finally {
            singleThreadedExecutor.shutdown();
        }
    }

    private static void delay(long time) {
        try {
            Thread.sleep(time);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}