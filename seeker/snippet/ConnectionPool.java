//date: 2023-11-10T17:07:02Z
//url: https://api.github.com/gists/b9a69b3810b1d0648f8ff564b1c25752
//owner: https://api.github.com/users/hsynercn

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

public class ConnectionPool {
    private static final int RESOURCE_RETRIEVAL_TRY_COUNT = 10;
    private static final int RESOURCE_RETRIEVAL_WAIT_PERIOD = 1000;
    private static final int POOL_SIZE = 10;
    private final Queue<Connection> connectionPool = new LinkedList<>();
    private int uninitializedConnectionCount;
    private String connectionParameter;
    private Exception createException;
    private final Object resourceRetrievalLock = new Object();

    public ConnectionPool(String connectionParameter) {
        this.connectionParameter = connectionParameter;

        List<Connection> connections = this.createResource(POOL_SIZE);
        synchronized (connectionPool) {
            this.connectionPool.addAll(connections);
            this.uninitializedConnectionCount = POOL_SIZE - connections.size();
        }
    }

    public final List<Connection> createResource(int resourceCount) {
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < resourceCount; i++) {
            list.add(i);
        }
        Exception resourceCreationExTemp = null;
        List<ConnCreationResult> createdConnectionsAll =  list.parallelStream()
                .map((Integer item) -> {
                    ConnCreationResult result = null;
                    try {
                        Connection newConnection = new Connection(this.connectionParameter);
                        result = new ConnCreationResult(newConnection, null);
                    } catch (Exception ex) {
                        //we store resource creation exceptions
                        result = new ConnCreationResult(null, ex);
                    }
                    return result;
                })
                .collect(Collectors.toList());

        LinkedList<Connection> resultList = new LinkedList<>();

        for (ConnCreationResult connCreationResult : createdConnectionsAll) {
            if (connCreationResult.getConnection() != null) {
                resultList.add(connCreationResult.getConnection());
            } else {
                resourceCreationExTemp = connCreationResult.getException();
            }
        }

        this.createException = resourceCreationExTemp;
        return resultList;
    }

    
    public synchronized boolean checkConnections() {
        try {
            int currentUninitializedConnectionCount = 0;
            synchronized (connectionPool) {
                LinkedList<Connection> aliveConnectionPool = new LinkedList<>();

                aliveConnectionPool.addAll(this.connectionPool);

                LinkedList<Connection> removeFromPool = new LinkedList<>();

                List<Connection> connectionPoolTempCopy = new ArrayList<>(aliveConnectionPool);
                for (Connection connection : connectionPoolTempCopy) {
                    if (!connection.healthcheck()) {
                        removeFromPool.add(connection);
                    }
                }

                this.uninitializedConnectionCount += removeFromPool.size();
                currentUninitializedConnectionCount = uninitializedConnectionCount;
                aliveConnectionPool.removeAll(removeFromPool);
                this.connectionPool.clear();
                this.connectionPool.addAll(aliveConnectionPool);
            }

            LinkedList<Connection> newConnectionPool = new LinkedList<>();
            if (currentUninitializedConnectionCount > 0) {
                List<Connection> newConnections = this.createResource(currentUninitializedConnectionCount);
                if (!newConnections.isEmpty()) {
                    newConnectionPool.addAll(newConnections);
                }
            }

            synchronized (connectionPool) {
                this.uninitializedConnectionCount -= newConnectionPool.size();
                this.connectionPool.addAll(newConnectionPool);
            }

        } catch (Exception ex) {
            return false;
        }

        return true;
    }

    public Connection getConnection() throws Exception {
        synchronized (resourceRetrievalLock) {
            int tryCount = RESOURCE_RETRIEVAL_TRY_COUNT;
            boolean isPoolEmpty = true;

            synchronized (connectionPool) {
                isPoolEmpty = connectionPool.isEmpty();
            }

            while (tryCount > 0 && isPoolEmpty && createException == null) {
                resourceRetrievalLock.wait(RESOURCE_RETRIEVAL_WAIT_PERIOD);
                synchronized (connectionPool) {
                    isPoolEmpty = connectionPool.isEmpty();
                }
                tryCount--;
            }
            synchronized (connectionPool) {
                if (!connectionPool.isEmpty()) {
                    this.uninitializedConnectionCount++;
                    return connectionPool.poll();
                } else {

                    Exception resultException = null;

                    if(this.createException != null) {
                        resultException = this.createException;
                    }

                    if(resultException == null) {
                        resultException = new Exception("Connection pool is empty");
                    }

                    throw resultException;
                }
            }
        }
    }
}

class Connection {
    private String connectionParameter;

    public Connection(String connectionParameter) {
        this.connectionParameter = connectionParameter;
    }

    public boolean healthcheck() {
        return true;
    }
}

class ConnCreationResult {
    private Connection connection;
    private Exception exception;

    public ConnCreationResult(Connection connection, Exception exception) {
        this.connection = connection;
        this.exception = exception;
    }
    
    public Connection getConnection() {
        return connection;
    }

    public Exception getException() {
        return exception;
    }
}