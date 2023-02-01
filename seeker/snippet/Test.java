//date: 2023-02-01T17:00:07Z
//url: https://api.github.com/gists/45c771a75ea0fe722f5ef5a263e2c748
//owner: https://api.github.com/users/stevenshan

import java.net.URI;
import java.util.ArrayList;

import software.amazon.awssdk.enhanced.dynamodb.DynamoDbAsyncTable;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedAsyncClient;
import software.amazon.awssdk.enhanced.dynamodb.Key;
import software.amazon.awssdk.enhanced.dynamodb.TableSchema;
import software.amazon.awssdk.enhanced.dynamodb.model.BatchGetItemEnhancedRequest;
import software.amazon.awssdk.enhanced.dynamodb.model.ReadBatch;
import software.amazon.awssdk.services.dynamodb.DynamoDbAsyncClient;

public class Test {
    public static void main(String[] args) {
        final DynamoDbEnhancedAsyncClient client = DynamoDbEnhancedAsyncClient
                .builder()
                .dynamoDbClient(DynamoDbAsyncClient
                        .builder()
                        .endpointOverride(URI.create("http://localhost:8000"))
                        .build())
                .build();

        final DynamoDbAsyncTable<DynamoObject> table = client.table("table", TableSchema.fromImmutableClass(DynamoObject.class));

        final var readBatch = ReadBatch
                .builder(DynamoObject.class)
                .mappedTableResource(table)
                .addGetItem(Key.builder().partitionValue("hello").build())
                .build();

        final var request = BatchGetItemEnhancedRequest
                .builder()
                .addReadBatch(readBatch)
                .build();

        while (true) {
            final ArrayList<DynamoObject> objects = new ArrayList<>();
            final var response = client.batchGetItem(request);

            response.resultsForTable(table).subscribe(objects::add).join();

            if (objects.size() != 1) {
                // this should never occur since the item is in the table
                throw new RuntimeException("failed: " + objects);
            }
        }
    }
}