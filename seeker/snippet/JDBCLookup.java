//date: 2022-06-03T16:44:19Z
//url: https://api.github.com/gists/33a920c712131121a34ebdb0d31dd6a1
//owner: https://api.github.com/users/kotnetrezviy

package namespace;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;

import com.sap.aii.mapping.api.AbstractTrace;
import com.sap.aii.mapping.api.StreamTransformationConstants;
import com.sap.aii.mapping.lookup.Channel;
import com.sap.aii.mapping.lookup.DataBaseAccessor;
import com.sap.aii.mapping.lookup.DataBaseResult;
import com.sap.aii.mapping.lookup.LookupService;

public class JDBCLookup {

    public static Node execute(String guid, String selectedColumns, String query, String service, String channelName,
        Map inputParam) throws Exception {

        AbstractTrace trace = (AbstractTrace) inputParam.get(StreamTransformationConstants.MAPPING_TRACE);

        Channel channel = null;
        DataBaseAccessor accessor = null;
        DataBaseResult resultSet = null;
        Node responseNode = null;

        String[] selectedColumnsArray = selectedColumns.split("\\s*,\\s*");
        for (int i = 0; i < selectedColumnsArray.length; i++) {
            if (selectedColumnsArray[i].contains(" as ")) {
                selectedColumnsArray[i] = selectedColumnsArray[i]
                    .substring(selectedColumnsArray[i].indexOf(" as ") + 4);
            }
        }

        trace.addDebugMessage("JDBC Statement: " + query);

        try {
            channel = LookupService.getChannel(service, channelName);
            accessor = LookupService.getDataBaseAccessor(channel);
            resultSet = accessor.execute(query);

            StringBuilder sb = new StringBuilder();
            sb.append("<?xml version=\"1.0\" encoding=\"utf-8\"?><Response>");

            for (Iterator rows = resultSet.getRows(); rows.hasNext();) {
                Map rowMap = (Map) rows.next();
                for (String columnName: selectedColumnsArray) {
                    sb.append("<").append(columnName).append(">").append((String) rowMap.get(columnName.toUpperCase()))
                        .append("</").append(columnName).append(">");
                }
            }
            sb.append("</Response>");

            InputStream responseStream = new ByteArrayInputStream(sb.toString().getBytes(StandardCharsets.UTF_8));
            TeeInputStream tee = new TeeInputStream(responseStream);

            DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
            Document document = db.parse(tee);

            trace.addDebugMessage("JDBC Response: " + tee.getStringContent());

            responseNode = document.getFirstChild();

            tee.close();
            responseStream.close();
        } catch (Throwable t) {
            StringWriter sw = new StringWriter();
            t.printStackTrace(new PrintWriter(sw));
            trace.addWarning(sw.toString());
        } finally {
            if (accessor != null)
                accessor.close();
        }
        return responseNode;
    }

    /**
     * Helper class which collects stream input while reading.
     */
    static class TeeInputStream extends InputStream {
        private ByteArrayOutputStream baos;
        private InputStream wrappedInputStream;

        TeeInputStream(InputStream inputStream) {
            baos = new ByteArrayOutputStream();
            wrappedInputStream = inputStream;
        }

        String getStringContent() {
            return baos.toString();
        }

        public int read() throws IOException {
            int r = wrappedInputStream.read();
            baos.write(r);
            return r;
        }
    }
}