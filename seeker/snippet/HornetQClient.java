//date: 2022-11-09T17:17:26Z
//url: https://api.github.com/gists/1854d970a284b984cafceb639af2d89c
//owner: https://api.github.com/users/MrGlass42

package hornetq;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import javax.jms.Connection;
import javax.jms.ConnectionFactory;
import javax.jms.JMSException;
import javax.jms.Topic;

import org.hornetq.api.core.TransportConfiguration;
import org.hornetq.api.jms.HornetQJMSClient;
import org.hornetq.api.jms.JMSFactoryType;
import org.hornetq.core.remoting.impl.netty.NettyConnectorFactory;

/**
 * HornetQClient
 * 
 * @author monzou
 */
class HornetQClient {

    private final AtomicReference<ConnectionFactory> connectionFactoryReference;

    HornetQClient() {
        connectionFactoryReference = new AtomicReference<>();
    }

    ConnectionFactory getConnectionFactory() {

        if (connectionFactoryReference.get() != null) {
            return connectionFactoryReference.get();
        }

        Map<String, Object> params = new HashMap<String, Object>();
        params.put("host", System.getProperty("hornetq.remoting.netty.host"));
        params.put("port", System.getProperty("hornetq.remoting.netty.port"));
        TransportConfiguration config = new TransportConfiguration(NettyConnectorFactory.class.getName(), params);
        ConnectionFactory factory = (ConnectionFactory) HornetQJMSClient.createConnectionFactoryWithoutHA(JMSFactoryType.CF, config);
        connectionFactoryReference.compareAndSet(null, factory);
        return connectionFactoryReference.get();

    }

    Connection connect() throws JMSException {
        Connection connection = getConnectionFactory().createConnection();
        connection.start();
        return connection;
    }

    Topic getExampleTopic() {
        return HornetQJMSClient.createTopic(System.getProperty("hornetq.topic.example.name"));
    }

}
