//date: 2023-06-19T16:40:11Z
//url: https://api.github.com/gists/79959e4cb568a928b6e035cb46b8ee19
//owner: https://api.github.com/users/happyhegde

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.agrona.concurrent.ManyToManyConcurrentArrayQueue;

import com.google.common.base.Charsets;

public class MessageProcessorMultiThreaded {
    
  private static Queue<Long> latencyQueue = new LinkedList<>();

  private static final int NUM_QUEUES = 64;
  private static final int QUEUE_CAPACITY = 30000;
  public static final int THREADS_PER_QUEUE = 1;
  public static final int BATCH_SIZE_PER_CALL = 1000;
  private AtomicBoolean isShutdown = new AtomicBoolean(false);

  private List<ManyToManyConcurrentArrayQueue<String>> queues;
  private ExecutorService executorService;


  public MessageProcessorMultiThreaded() {
    queues = new ArrayList<>();
    executorService = Executors.newFixedThreadPool(THREADS_PER_QUEUE * NUM_QUEUES);
    for (int i = 0; i < NUM_QUEUES; i++) {
      queues.add(i, new ManyToManyConcurrentArrayQueue<String>(QUEUE_CAPACITY));
      submitTasksForQueue(i);
    }
  }

  private void submitTasksForQueue(int indexQueue) {
    for (int i = 0; i < THREADS_PER_QUEUE; i++) {
      executorService.submit(() -> {
        while (!isShutdown.get()) {
          Set<String> entries = new HashSet<>();
          String idfvLogEntry = null;
          while (true) {
            String entry = this.queues.get(indexQueue).poll();
            if (entry != null) {
              entries.add(idfvLogEntry);
              if (entries.size() == BATCH_SIZE_PER_CALL) {
                long t1 = System.currentTimeMillis();
                process(entries);
                entries.clear();
                long t2 = System.currentTimeMillis();
                recordLatency(t2 - t1);
              }
            } else {
              if (isShutdown.get()) {
                process(entries);
                entries.clear();
                break;
              } else {
                try {
                  Thread.sleep(10);
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                }
              }
            }
          }
        }
      });
    }
  }

  private void process(Set<String> messages) {
    for (String message : messages) {
      // do the operation here.
    }
  }

  public void enqueue(String message) {
    int index = getIndex(message);
    boolean isSuccess = false;
    while (!isSuccess) {
      isSuccess = this.queues.get(index).offer(message);
    }
  }

  private int getIndex(String message) {
    long keyHashed = Constant.hf.newHasher()
        .putString(message, Charsets.UTF_8)
        .hash().asLong();
    return Math.abs((int) (keyHashed % NUM_QUEUES));
  }

  public synchronized Map<Integer, Integer> countQueueReads() {
    Map<Integer, Integer> queueCount = new HashMap<>();
    AtomicInteger i = new AtomicInteger(0);
    try {
      queues.forEach(a -> queueCount.put(i.getAndIncrement(), a.size()));
    } catch (Exception ignored) {}
    return queueCount;
  }

  private synchronized void recordLatency(long ms) {
    while (latencyQueue.size() > 10000) {
      latencyQueue.remove();
    }
    latencyQueue.add(ms);
  }

  public static double avgLatency() {
    Queue<Long> longs = new LinkedList<>(latencyQueue);
    double latencySum = 0.0;
    int i = 1;
    while (longs.peek() != null) {
      latencySum += (double) longs.poll();
      i++;
    }
    return latencySum / i / BATCH_SIZE_PER_CALL;
  }


  public void shutdown() {
    isShutdown.set(true);
  }
}
