//date: 2024-01-03T16:48:53Z
//url: https://api.github.com/gists/185bbe21c97189b72cfa01587539dfdb
//owner: https://api.github.com/users/SA-JackMax

import org.alfresco.jlan.server.SrvSession;
import org.alfresco.jlan.server.core.DeviceContext;
import org.alfresco.jlan.server.filesys.FileExistsException;
import org.alfresco.jlan.server.filesys.FileName;
import org.alfresco.jlan.server.filesys.FileOpenParams;
import org.alfresco.jlan.server.filesys.NetworkFile;
import org.alfresco.jlan.server.filesys.TreeConnection;
import org.alfresco.jlan.smb.server.disk.JavaFileDiskDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

/**
 * Created by dkopel on 3/13/16.
 */

@Component
public class InMemoryFileDriver extends JavaFileDiskDriver {

    private Logger logger = LoggerFactory.getLogger(getClass());

    private NetworkFile f;

    @Override
    public NetworkFile createFile(SrvSession sess, TreeConnection tree, FileOpenParams params) throws IOException {
        DeviceContext ctx = tree.getContext();
        String fname = FileName.buildPath(ctx.getDeviceName(), params.getPath(), (String)null, File.separatorChar);
        File file = new File(fname);

        logger.info("User {} created a new file called {}.", sess.getClientInformation().getUserName(), fname);

        if(file.exists()) {
            throw new FileExistsException();
        } else {
            InMemoryNetworkFile netFile = new InMemoryNetworkFile(fname);
            netFile.setGrantedAccess(2);
            netFile.setAllowedAccess(2);
            netFile.setFullName(params.getPath());
            return netFile;
        }
    }

    @Override
    public int writeFile(SrvSession sess, TreeConnection tree, NetworkFile file, byte[] buf, int bufoff, int siz, long fileoff) throws IOException {
        logger.info("Write request to file {} with data {} {}.", file.getFullName(), siz);
        return super.writeFile(sess, tree, file, buf, bufoff, siz, fileoff);
    }

    @Override
    public void closeFile(SrvSession sess, TreeConnection tree, NetworkFile file) throws IOException {
        super.closeFile(sess, tree, file);
        logger.info("Close file {}.", file.getFullName(), sess.getClientInformation().getUserName());
        
        
        // Over here if we had a write we can now transfer the contents of the file into our system!
    }
}
