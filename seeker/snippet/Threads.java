//date: 2025-03-26T17:06:22Z
//url: https://api.github.com/gists/ddd3fd585edb4e7a0da7b121232d0483
//owner: https://api.github.com/users/barcellos-pedro

import java.util.List;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/// Threads Demo
public class Main {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println(getSystemInfo());

        /// Manual Thread handling
        ThreadFactory threadFactory = Executors.defaultThreadFactory();

        Thread thread = threadFactory.newThread(() -> {
            try {
                System.out.println(Thread.currentThread());
                Thread.sleep(3_000L);
                System.out.println("Thread task finished");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        thread.start();
        System.out.println("Waiting for thread to finish");
        thread.join(); // blocks here
        System.out.println("Manual Threads Done\n");

        /// -------------------------------------------------------------------------------

        /// Executor's Thread handling (Uses Pooling)
        ExecutorService executor = Executors.newFixedThreadPool(getCpus());

        List<Callable<String>> jobs = getJobsToProcess();

        Callable<String> task = () -> Thread.currentThread() + " | Hello from callable-task-1";

        Future<String> futureResult = executor.submit(task);

        executor.submit(() -> System.out.println(Thread.currentThread() + " | hello from task-2"));

        System.out.println("Is callable task done? " + (futureResult.isDone() ? "Yes" : "No"));

        String result = futureResult.get(); // blocks here

        System.out.println("Is callable task done? " + (futureResult.isDone() ? "Yes" : "No"));

        System.out.println(result);

        executor.invokeAll(jobs)
                .stream()
                .map(Main::getJobResult)
                .forEach(System.out::println);

        executor.shutdown();
        executor.close();

        System.out.println("Executor Done");
    }

    private static String getJobResult(Future<String> future) {
        try {
            return future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<Callable<String>> getJobsToProcess() {
        return IntStream.range(0, 8).mapToObj(id -> (Callable<String>) () -> {
            try {
                if (id == 3 || id == 6) throw new RuntimeException(id + "# task has errors");

                System.out.println(Thread.currentThread() + "Working on task #" + id);
                Thread.sleep(1_000L);
                System.out.println(Thread.currentThread() + "Done task #" + id);
                
                return id + "# Work OK";
            } catch (RuntimeException exception) {
                return id + "# Work failed";
            }
        }).toList();
    }

    public static int getCpus() {
        return Runtime.getRuntime().availableProcessors();
    }

    public static String getSystemInfo() {
        return String.format("CPUs: %s | Thread: %s | ID: %s | Count: %s",
                Runtime.getRuntime().availableProcessors(),
                Thread.currentThread().getName(),
                Thread.currentThread().threadId(),
                Thread.activeCount());
    }
}