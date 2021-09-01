//date: 2021-09-01T13:03:44Z
//url: https://api.github.com/gists/d9c5eb448160557b116ccb902d157b0d
//owner: https://api.github.com/users/abstractdog

package org.apache.tez.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.tez.dag.api.records.DAGProtos;
import org.xerial.snappy.SnappyInputStream;

import com.google.common.io.Files;
import com.google.common.primitives.Bytes;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;

public class TestPayloadParser {

  public static void main(String[] args) throws IOException {

    /*
      FIRST: get a text file representing bytes of a user payload from a heap dump with VisualVM, use OQL script:
      
      var writer = new FileWriterClass("/tmp/bytes.txt");
      var found = false;
      
      map(heap.objects("org.apache.tez.common.ContainerTask"), 
        function (obj, index, array, result) {
          if (found) {
              return null;
          }
          if (obj.taskSpec == null){
              return null;
          }
          found = true;
          var array = obj.taskSpec.processorDescriptor.userPayload.payload.hb;
          for (var i = 0; i < array.length; i++) {
              writer.append(array[i] + ",");
          }
          writer.close();
       })
     */
    BufferedReader bufferedReader =
        new BufferedReader(new InputStreamReader(new FileInputStream(new File("/tmp/bytes.txt"))));
    List<Byte> listOfBytes = new ArrayList<>();
    int r;

    StringBuilder b = new StringBuilder();
    while ((r = bufferedReader.read()) != -1) {
      char ch = (char) r;
      if (ch == ',') {
        listOfBytes.add((byte) Integer.parseInt(b.toString()));
        b.setLength(0);
      } else {
        b.append(ch);
      }
    }
    bufferedReader.close();
    System.out.println("finished read");
    byte[] bytes = Bytes.toArray(listOfBytes);
    System.out.println("finished bytes");
    ByteBuffer payload = ByteBuffer.wrap(bytes);
    System.out.println("wrapped bytes");
    ByteString byteString = ByteString.copyFrom(payload);

    Configuration configuration = createConfFromByteString(byteString);

    System.out.println(configuration);
    Map<String, String> mapConfig = new HashMap<String, String>();
    Iterator<Map.Entry<String, String>> iterator = configuration.iterator();
    while (iterator.hasNext()) {
      Map.Entry<String, String> entry = iterator.next();
      mapConfig.put(entry.getKey(), entry.getValue());
    }
    Files.write(mapConfig.toString().getBytes(), new File("/tmp/config.txt"));
  }

  public static Configuration createConfFromByteString(ByteString byteString) throws IOException {
    try (SnappyInputStream uncompressIs = new SnappyInputStream(byteString.newInput())) {
      CodedInputStream in = CodedInputStream.newInstance(uncompressIs);
      in.setSizeLimit(Integer.MAX_VALUE);
      DAGProtos.ConfigurationProto confProto = DAGProtos.ConfigurationProto.parseFrom(in);
      Configuration conf = new Configuration(false);
      readConfFromPB(confProto, conf);
      return conf;
    }
  }

  private static void readConfFromPB(DAGProtos.ConfigurationProto confProto, Configuration conf) {
    List<DAGProtos.PlanKeyValuePair> settingList = confProto.getConfKeyValuesList();
    for (DAGProtos.PlanKeyValuePair setting : settingList) {
      conf.set(setting.getKey(), setting.getValue());
    }
  }
}
