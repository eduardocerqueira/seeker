//date: 2022-09-07T16:59:41Z
//url: https://api.github.com/gists/63db3ab617421f2c70a1d88cad91b3f5
//owner: https://api.github.com/users/nareshmaharaj-consultant

import com.aerospike.client.*;
import com.aerospike.client.Record;
import com.aerospike.client.policy.ClientPolicy;
import com.aerospike.client.policy.RecordExistsAction;
import com.aerospike.client.policy.ScanPolicy;
import com.aerospike.client.policy.WritePolicy;
import java.util.UUID;

public class GetKeyTTL {

    public AerospikeClient client = setUpConnection();

    public GetKeyTTL() throws Exception {
        WritePolicy wPolicy;
        wPolicy = new WritePolicy();
        wPolicy.recordExistsAction = RecordExistsAction.UPDATE;
        wPolicy.sendKey = true;
        wPolicy.expiration = 600;
        String namespace = "test";
        String set = "keyttl";

        for ( int i=0; i<600; i++){
            Bin bin = new Bin("itemID",  UUID.randomUUID().toString() );
            Key key = new Key(namespace, set,  String.valueOf(i));
            wPolicy.expiration = 600-i;
            client.put(wPolicy, key, bin);
        }

        ScanPolicy policy = new ScanPolicy();
        client.scanAll(policy, namespace, set, new GetKeyTTL.ScanParallel());
    }

    public AerospikeClient setUpConnection() throws Exception {
        String host = "localhost";
        int port = 3000;

        Host[] hosts = new Host[] {
                new Host( host, port)
        };

        ClientPolicy clientPolicy = new ClientPolicy();
        AerospikeClient client = new AerospikeClient(clientPolicy, hosts);
        return client;
    }

    public class ScanParallel implements ScanCallback {
        public void scanCallback(Key key, Record record) {
            long ttl = record.getTimeToLive();
            String userKey = key.userKey.toString();
            System.out.format("{key:\"%s\", ttl:%s}\n", userKey, ttl);
        }
    }

    public static void main(String [] args) throws Exception {
        GetKeyTTL keyTTL = new GetKeyTTL();
    }
}