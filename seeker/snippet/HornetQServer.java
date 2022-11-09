//date: 2022-11-09T17:17:26Z
//url: https://api.github.com/gists/1854d970a284b984cafceb639af2d89c
//owner: https://api.github.com/users/MrGlass42

package hornetq;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.hornetq.jms.server.embedded.EmbeddedJMS;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HornetQServer
 * 
 * @author monzou
 */
class HornetQServer extends HornetQExampleBase {

    private static final Logger LOGGER = LoggerFactory.getLogger(HornetQServer.class);

    public static void main(String[] args) {
        new HornetQServer().start();
    }

    private final Lock lock;

    private final EmbeddedJMS server;

    HornetQServer() {
        lock = new ReentrantLock(false);
        server = new EmbeddedJMS();
    }

    void start() {
        lock.lock();
        try {
            server.start();
        } catch (Exception e) {
            throw new RuntimeException("Failed to start JMS server", e);
        } finally {
            lock.unlock();
        }
    }

    void sotp() {
        lock.lock();
        try {
            try {
                server.stop();
            } catch (Exception e) {
                LOGGER.warn("Failed to stop JMS server", e);
            }
        } finally {
            lock.unlock();
        }
    }

}
