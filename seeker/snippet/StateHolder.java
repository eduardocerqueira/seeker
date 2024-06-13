//date: 2024-06-13T16:58:16Z
//url: https://api.github.com/gists/895a2e8d068271b6059bc3f8243ddfd6
//owner: https://api.github.com/users/adamciolkowski

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class StateHolder {

    public enum State {
        ALLOWED,
        NOT_ALLOWED,
        NOT_ESTABLISHED
    }

    private final Lock lock = new ReentrantLock();
    private final Condition established = lock.newCondition();
    private State state = State.NOT_ESTABLISHED;

    public void setState(State state) {
        lock.lock();
        try {
            this.state = state;
            established.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public State waitUntilEstablished() throws InterruptedException {
        lock.lock();
        try {
            while (state == State.NOT_ESTABLISHED) {
                established.await();
            }
            return state;
        } finally {
            lock.unlock();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        StateHolder unnamed = new StateHolder();

        Runnable runnable = () -> {
            try {
                String name = Thread.currentThread().getName();
                System.out.println(name + " waiting...");
                State state = unnamed.waitUntilEstablished();
                System.out.println(name + " done: " + state);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        };
        new Thread(runnable).start();
        Thread.sleep(1_000);
        new Thread(runnable).start();
        Thread.sleep(1_000);
        new Thread(runnable).start();

        Thread.sleep(1_000);
        unnamed.setState(State.ALLOWED);
    }
}
