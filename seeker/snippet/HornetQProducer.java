//date: 2022-11-09T17:17:26Z
//url: https://api.github.com/gists/1854d970a284b984cafceb639af2d89c
//owner: https://api.github.com/users/MrGlass42

package hornetq;

import java.util.Date;

import javax.jms.JMSException;
import javax.jms.MessageProducer;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HornetQProducer
 * 
 * @author monzou
 */
class HornetQProducer extends HornetQExampleBase {

    private static final Logger LOGGER = LoggerFactory.getLogger(HornetQProducer.class);

    public static void main(String[] args) {
        new HornetQProducer().start();
    }
    
    private final HornetQClient client;

    HornetQProducer() {
        client = new HornetQClient();
    }

    void start() {
        try {
            Session session = createSession();
            MessageProducer producer = session.createProducer(client.getExampleTopic());
            new Thread(new Ping(session, producer)).start();
        } catch (JMSException e) {
            throw new RuntimeException("Failed to start session", e);
        }
    }

    private Session createSession() throws JMSException {
        return client.connect().createSession(false, Session.AUTO_ACKNOWLEDGE);
    }

    private static class Ping implements Runnable {

        final Session session;

        final MessageProducer producer;

        Ping(Session session, MessageProducer producer) {
            this.session = session;
            this.producer = producer;
        }

        /** {@inheritDoc} */
        @Override
        public void run() {
            do {
                try {
                    TextMessage message = session.createTextMessage("Ping: " + new Date());
                    producer.send(message);
                    Thread.sleep(1000);
                } catch (JMSException e) {
                    LOGGER.warn("Failed to send a message", e);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            } while (true);
        }

    }

}
