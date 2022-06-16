//date: 2022-06-16T16:56:57Z
//url: https://api.github.com/gists/a5c6a65a28f6e478a0eea34f0ad502f7
//owner: https://api.github.com/users/bryanck

package io.tabular.metadata.example;

import com.google.common.collect.ImmutableMap;
import java.nio.ByteOrder;
import java.util.Map;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Table;
import org.apache.iceberg.aws.s3.S3FileIO;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.rest.RESTCatalog;

public class Main {

  public static void main(String[] args) throws Exception {
    Map<String, String> catalogProps = ImmutableMap.of(
        "credential", "<creds>",
        "uri", "http://localhost:8088/ws",
        "io-impl", S3FileIO.class.getName() // optional, if you don't want to include Hadoop libraries
    );
    try (RESTCatalog catalog = new RESTCatalog()) {
      catalog.initialize("iceberg", catalogProps);
      Table table = catalog.loadTable(TableIdentifier.parse("default.tester"));

      Expression filter = Expressions.greaterThanOrEqual("dateint", 20220615);
      filter = Expressions.and(filter, Expressions.lessThan("dateint", 20220616));

      int slotColId = table.schema().findField("slot").fieldId();

      try (CloseableIterable<FileScanTask> tasks = table.newScan()
          .filter(filter)
          .includeColumnStats()
          .planFiles()) {

        tasks.forEach(task -> {
          DataFile file = task.file();
          int lowerSlot = file.lowerBounds().get(slotColId).order(ByteOrder.LITTLE_ENDIAN).getInt();
          int upperSlot = file.upperBounds().get(slotColId).order(ByteOrder.LITTLE_ENDIAN).getInt();
          System.out.printf("file: %s, lower slot: %s, upper slot: %s%n", file.path(), lowerSlot,
              upperSlot);
        });
      }
    }
  }

}
