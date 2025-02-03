//date: 2025-02-03T16:54:40Z
//url: https://api.github.com/gists/f631a91f33467999a2fb9a9e6ac883a5
//owner: https://api.github.com/users/scottf

import io.nats.client.*;
import io.nats.client.api.AckPolicy;
import io.nats.client.api.ConsumerConfiguration;
import io.nats.client.api.ConsumerInfo;
import io.nats.client.impl.ErrorListenerConsoleImpl;
import io.nats.client.impl.NatsJetStreamMetaData;

import java.io.IOException;
import java.util.List;

import static io.nats.client.support.JsonUtils.getFormatted;

public class LargeSimplfiedFetchSlowConsumer {

    static String URL = "js:js@vnd.east.nats.dev";
    static String STREAM = "test-limits";
    static String CONSUMER_PREFIX = "nv-java-";
    static final int MAX_ACK_PENDING = 65536;
    static final int MAX_MESSAGES = 1000;
    static final int MAX_BYTES = 32 * 1024 * 1024;
    static final long CONSUMER_CHECK_TIME = 5000; // every second

    public static void main(String[] args) {
        Options options = Options.builder()
            .server(URL)
            .connectionListener((conn, event) -> System.out.println("CL: " + event))
            .errorListener(new ErrorListenerConsoleImpl())
            .build();
        try (Connection nc = Nats.connect(options)) {
            System.out.println("CONNECTED: " + nc.getServerInfo());
            JetStreamManagement jsm = nc.jetStreamManagement();
            JetStream js = nc.jetStream();

            removeExistingTestConsumers(jsm);

            // MAKE A NEW CONSUMER TO TRY
            String consumerName = CONSUMER_PREFIX + NUID.nextGlobalSequence(); // random suffix
            ConsumerConfiguration cc = ConsumerConfiguration.builder()
                .durable(consumerName)
                .maxAckPending(MAX_ACK_PENDING)
                .ackPolicy(AckPolicy.Explicit)
                .build();

            StreamContext sc = js.getStreamContext(STREAM);
            ConsumerContext ctx = sc.createOrUpdateConsumer(cc);
            System.out.println("CONSUMER: " + getFormatted(ctx.getCachedConsumerInfo().getConsumerConfiguration()));
            Thread checkConsumerThread = new Thread(() -> checkConsumer(consumerName));
            checkConsumerThread.start();

            FetchConsumeOptions fco = FetchConsumeOptions.builder()
                .max(MAX_BYTES, MAX_MESSAGES)
                .build();
            while (true) {
                System.out.println("START FETCH: " + fco.toJson());
                long bytes = 0;
                FetchConsumer fc = ctx.fetch(fco);
                while (!fc.isFinished()) {
                    Message msg = fc.nextMessage();
                    if (msg == null) {
                        System.out.println("FETCH: No message. Finished? " + fc.isFinished());
                        Thread.sleep(50);
                    }
                    else {
                        long cbc = msg.consumeByteCount();
                        bytes += cbc;
                        System.out.println("FETCHED: (" + cbc + "/" + bytes + ") " + toString(msg));
                        msg.ack();
                    }
                }
            }

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    static void checkConsumer(String consumerName) {
        boolean sleep = true;
        Options options = Options.builder()
            .server(URL)
            .connectionListener((conn, event) -> System.out.println("CL: " + event))
            .errorListener(new ErrorListenerConsoleImpl())
            .build();
        try (Connection nc = Nats.connect(options)) {
            JetStreamManagement jsm = nc.jetStreamManagement();
            while (true) {
                try {
                    Thread.sleep(CONSUMER_CHECK_TIME);
                    ConsumerInfo ci = jsm.getConsumerInfo(STREAM, consumerName);
                    System.out.println("CHECK CONSUMER: " + toString(ci));
                }
                catch (IOException | JetStreamApiException e) {
                    System.out.println("CHECK CONSUMER EX: " + e.getMessage());
                }
                catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static void removeExistingTestConsumers(JetStreamManagement jsm) throws IOException, JetStreamApiException {
        List<ConsumerInfo> ciList = jsm.getConsumers(STREAM);
        for (ConsumerInfo ci : ciList) {
            if (ci.getName().startsWith(CONSUMER_PREFIX)) {
                jsm.deleteConsumer(STREAM, ci.getName());
            }
        }
    }

    private static String toString(Message msg) {
        NatsJetStreamMetaData meta = msg.metaData();
        return "StreamSeq: " + meta.streamSequence() + " | "
            + "ConSeq: " + meta.consumerSequence() + " | "
            + "Delivered: " + meta.deliveredCount() + " | "
            + "Pending: " + meta.pendingCount();
    }

    private static String toString(ConsumerInfo ci) {
        return "Waiting: " + ci.getNumWaiting() + " | "
            + "Delivered: " + ci.getDelivered() + " | "
            + "Redelivered: " + ci.getRedelivered() + " | "
            + "Pending: " + ci.getNumAckPending();
    }
}