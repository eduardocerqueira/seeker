//date: 2022-02-09T17:00:36Z
//url: https://api.github.com/gists/15c082d451c6ecc04eece058656b9f7a
//owner: https://api.github.com/users/absurdfarce

package com.datastax.driver;

import com.datastax.driver.core.Cluster;
import com.datastax.driver.core.ResultSet;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;
import com.datastax.shaded.guava.util.concurrent.Futures;
import com.datastax.shaded.guava.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.FutureCallback;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;

public class KeyspaceCount {

    private static class KCListener implements FutureCallback<ResultSet> {

        @Override
        public void onSuccess(ResultSet rs) {

            Row row = rs.one();
            System.out.println(String.format("Answer: %s keyspaces", row.getLong("count")));
        }

        @Override
        public void onFailure(Throwable t) {}
    }

    public static void main(String[] args) {

        InetAddress localhost = null;
        try { localhost = InetAddress.getByName("127.0.0.1"); }
        catch (UnknownHostException uhe) {

            System.out.println("Couldn't find localhost!");
        }

        Cluster cluster = null;
        try {
            cluster =
                    Cluster.builder()
                    .addContactPointsWithPorts(new InetSocketAddress(localhost, 9042))
                    .build();
            Session session = cluster.connect();
            ListenableFuture<ResultSet> rs = session.executeAsync("select count(*) from system_schema.keyspaces");

            // This fails to compile.
            //
            // KCListener implements com.google.common.util.concurrent.FutureCallback but we're dealing with
            // com.datastax.shaded.guava.util.concurrent.Futures which expects to get an arg of type
            // com.datastax.shaded.guava.util.concurrent.FutureCallback.
            Futures.addCallback(rs, new KCListener());
        }
        finally {
            cluster.close();
        }
    }
}