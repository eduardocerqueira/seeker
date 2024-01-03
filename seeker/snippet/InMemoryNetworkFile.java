//date: 2024-01-03T16:48:53Z
//url: https://api.github.com/gists/185bbe21c97189b72cfa01587539dfdb
//owner: https://api.github.com/users/SA-JackMax

import org.alfresco.jlan.server.filesys.NetworkFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author Created by dkopel on 3/15/16.
 */
public class InMemoryNetworkFile extends NetworkFile {

    private ByteArrayOutputStream out = new ByteArrayOutputStream();

    private byte[] data;

    private AtomicBoolean opened = new AtomicBoolean(false);

    private Logger logger = LoggerFactory.getLogger(getClass());

    public InMemoryNetworkFile(int fid) {
        super(fid);
    }

    public InMemoryNetworkFile(int fid, int did) {
        super(fid, did);
    }

    public InMemoryNetworkFile(int fid, int stid, int did) {
        super(fid, stid, did);
    }

    public InMemoryNetworkFile(String name) {
        super(name);
    }

    @Override
    public void openFile(boolean b) throws IOException {
        opened.set(b);
    }

    @Override
    public int readFile(byte[] bytes, int i, int i1, long l) throws IOException {
        return 0;
    }

    @Override
    public void writeFile(byte[] bytes, int len, int pos, long offset) throws IOException {
        logger.debug("Bytes Size: {}, Length: {}, Position: {}, Offset: {}", bytes.length, len, pos, offset);
        out.write(bytes, pos, len);
        logger.debug("Data is now {} bytes", out.size());
    }

    @Override
    public long seekFile(long l, int i) throws IOException {
        return 0;
    }

    @Override
    public void flushFile() throws IOException {
        logger.info("Flushed!");
    }

    @Override
    public void truncateFile(long l) throws IOException {
        out.reset();
    }

    @Override
    public void closeFile() throws IOException {
        opened.set(false);
        out.close();
        data = out.toByteArray();
    }

    public byte[] getData() {
        return data;
    }
}