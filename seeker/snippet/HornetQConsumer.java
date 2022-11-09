//date: 2022-11-09T17:17:26Z
//url: https://api.github.com/gists/1854d970a284b984cafceb639af2d89c
//owner: https://api.github.com/users/MrGlass42

package hornetq;

import javax.jms.JMSException;
import javax.jms.Message;
import javax.jms.MessageConsumer;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HornetQConsumer
 * 
 * @author monzou
 */
class HornetQConsumer extends HornetQExampleBase {

    private static final Logger LOGGER = LoggerFactory.getLogger(HornetQConsumer.class);

    public static void main(String[] args) {
        new HornetQConsumer().start();
    }

    private final HornetQClient client;

    HornetQConsumer() {
        client = new HornetQClient();
    }

    void start() {
        try {
            Receiver receiver = new Receiver(createSession().createConsumer(client.getExampleTopic()));
            new Thread(receiver).start();
        } catch (JMSException e) {
            throw new RuntimeException("Failed to establish connection", e);
        }
    }

    private Session createSession() throws JMSException {
        return client.connect().createSession(false, Session.AUTO_ACKNOWLEDGE);
    }

    private static class Receiver implements Runnable {

        final MessageConsumer consumer;

        Receiver(MessageConsumer consumer) {
            this.consumer = consumer;
        }

        /** {@inheritDoc} */
        @Override
        public void run() {
            do {
                try {
                    Message message = consumer.receive(0);
                    if (message != null) {
                        System.out.println("Received message: " + ((TextMessage) message).getText());
                    }
                } catch (JMSException e) {
                    LOGGER.warn("Failed to receive message", e);
                }
            } while (true);
        }

    }

}
