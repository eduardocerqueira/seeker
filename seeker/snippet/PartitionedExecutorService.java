//date: 2026-01-09T17:19:19Z
//url: https://api.github.com/gists/e840ebef0212e9180d4790aab2b7fcc9
//owner: https://api.github.com/users/neherim

package com.github.neherim.process.engine.core.utils.thread;

import java.time.Duration;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Класс-обертка над ExecutorService, обеспечивающий последовательное выполнение задач с одинаковым ключом партиции.
 * <p>
 * Гарантии:
 * <ul>
 *   <li>Задачи с одинаковым partitionKey выполняются строго последовательно в порядке поступления (FIFO)</li>
 *   <li>Задачи с разными partitionKey могут выполняться параллельно</li>
 *   <li>Thread-safe для конкурентного вызова submit() из разных потоков</li>
 * </ul>
 * <p>
 * Пример использования:
 * <pre>{@code
 * ExecutorService delegate = Executors.newFixedThreadPool(10);
 * PartitionedExecutorService<String> partitioned = new PartitionedExecutorService<>(delegate);
 *
 * // Задачи с ключом "user-123" выполнятся последовательно
 * partitioned.submit("user-123", () -> updateUserBalance());
 * partitioned.submit("user-123", () -> sendNotification());
 * partitioned.submit("user-123", () -> logActivity());
 *
 * // Задачи с ключом "user-456" выполняются параллельно с "user-123"
 * partitioned.submit("user-456", () -> updateUserBalance());
 * }</pre>
 *
 * @param <T> тип ключа партиции
 */
public class PartitionedExecutorService<T> {
    private final ExecutorService delegate;
    private final ConcurrentHashMap<T, PartitionQueue> partitions = new ConcurrentHashMap<>();
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    /**
     * Создает новый экземпляр PartitionedExecutorService.
     *
     * @param delegate ExecutorService для выполнения задач
     * @throws IllegalArgumentException если delegate == null
     */
    public PartitionedExecutorService(ExecutorService delegate) {
        if (delegate == null) {
            throw new IllegalArgumentException("delegate ExecutorService cannot be null");
        }
        this.delegate = delegate;
    }

    /**
     * Отправляет задачу на выполнение с указанным ключом партиции.
     * Задачи с одинаковым ключом будут выполняться последовательно в порядке поступления.
     *
     * @param partitionKey ключ партиции для группировки задач
     * @param task         задача для выполнения
     * @return CompletableFuture, который завершится после выполнения задачи
     * @throws IllegalArgumentException если partitionKey или task == null
     * @throws IllegalStateException    если ExecutorService уже shutdown
     */
    public CompletableFuture<Void> submit(T partitionKey, Runnable task) {
        Objects.requireNonNull(partitionKey, "partitionKey cannot be null");
        Objects.requireNonNull(task, "task cannot be null");

        if (isShutdown.get()) {
            throw new IllegalStateException("ExecutorService is shutdown");
        }

        // Получаем или создаем очередь для данной партиции
        PartitionQueue queue = partitions.computeIfAbsent(partitionKey, k -> new PartitionQueue());

        // Создаем CompletableFuture для результата
        CompletableFuture<Void> future = new CompletableFuture<>();

        // Добавляем задачу в очередь
        queue.enqueue(task, future);

        // Пытаемся запустить обработку очереди
        tryProcessNext(partitionKey, queue);

        return future;
    }

    /**
     * Пытается запустить следующую задачу из очереди партиции.
     * Если партиция уже обрабатывается, метод ничего не делает.
     *
     * @param partitionKey ключ партиции
     * @param queue        очередь задач партиции
     */
    private void tryProcessNext(T partitionKey, PartitionQueue queue) {
        // Пытаемся установить флаг обработки (CAS операция)
        if (!queue.isProcessing.compareAndSet(false, true)) {
            // Партиция уже обрабатывается, выходим
            return;
        }

        // Берем следующую задачу из очереди
        TaskWrapper wrapper = queue.taskQueue.poll();

        if (wrapper == null) {
            // Очередь пуста, сбрасываем флаг и удаляем партицию
            queue.isProcessing.set(false);
            // Удаляем партицию из карты, если очередь пуста
            // Используем remove с условием, чтобы не удалить партицию, если в нее добавили новую задачу
            partitions.computeIfPresent(partitionKey, (k, v) -> v.taskQueue.isEmpty() ? null : v);
            return;
        }

        // Запускаем задачу на делегированном executor
        CompletableFuture.runAsync(() -> {
            try {
                wrapper.task.run();
                wrapper.future.complete(null);
            } catch (Throwable ex) {
                wrapper.future.completeExceptionally(ex);
            }
        }, delegate).whenComplete((result, error) -> {
            // После завершения задачи сбрасываем флаг обработки
            queue.isProcessing.set(false);
            // И пытаемся запустить следующую задачу
            tryProcessNext(partitionKey, queue);
        });
    }

    /**
     * Завершает работу ExecutorService.
     * После вызова этого метода новые задачи будут отклоняться.
     * Активные задачи будут завершены с таймаутом.
     */
    public void shutdown() {
        isShutdown.set(true);
        ThreadUtils.gracefulShutdown(delegate, Duration.ofSeconds(10));
    }

    /**
     * Проверяет, завершен ли сервис.
     *
     * @return true если сервис shutdown, иначе false
     */
    public boolean isShutdown() {
        return isShutdown.get();
    }

    /**
     * Возвращает количество активных партиций (с задачами в очереди или выполняющимися).
     *
     * @return количество активных партиций
     */
    public int getPartitionCount() {
        return partitions.size();
    }

    /**
     * Возвращает размер очереди для указанного ключа партиции.
     *
     * @param partitionKey ключ партиции
     * @return размер очереди или 0, если партиции не существует
     */
    public int getQueueSize(T partitionKey) {
        PartitionQueue queue = partitions.get(partitionKey);
        return queue != null ? queue.taskQueue.size() : 0;
    }

    /**
     * Возвращает общее количество задач в очередях всех партиций.
     * Не включает задачи, которые уже выполняются.
     *
     * @return общее количество задач в очередях
     */
    public int getTotalQueuedTasks() {
        return partitions.values().stream()
                .mapToInt(queue -> queue.taskQueue.size())
                .sum();
    }

    /**
     * Внутренний класс для управления очередью задач одной партиции.
     */
    private static class PartitionQueue {
        private final Queue<TaskWrapper> taskQueue = new ConcurrentLinkedQueue<>();
        private final AtomicBoolean isProcessing = new AtomicBoolean(false);

        void enqueue(Runnable task, CompletableFuture<Void> future) {
            taskQueue.add(new TaskWrapper(task, future));
        }
    }

    /**
     * Обертка для задачи и ее CompletableFuture.
     */
    private static class TaskWrapper {
        private final Runnable task;
        private final CompletableFuture<Void> future;

        TaskWrapper(Runnable task, CompletableFuture<Void> future) {
            this.task = task;
            this.future = future;
        }
    }
}
