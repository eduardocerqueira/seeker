//date: 2024-04-26T17:06:52Z
//url: https://api.github.com/gists/64e6b695124075763e5f1ba4d4d191d5
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.zip.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class Main
{
    public static void main(String[] args) throws Exception // Convert rar to zip using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");

        try (com.aspose.zip.Archive zip = new com.aspose.zip.Archive())
        {
            try (com.aspose.zip.RarArchive rar = new com.aspose.zip.RarArchive("input.rar"))
            {
                for (int i = 0; i < rar.getEntries().size(); i++)
                {
                    com.aspose.zip.RarArchiveEntry entry = rar.getEntries().get(i);
                    if (!entry.isDirectory())
                    {
                        try (ByteArrayOutputStream out = new ByteArrayOutputStream())
                        {
                            entry.extract(out);
                            try (java.io.ByteArrayInputStream in = new java.io.ByteArrayInputStream(out.toByteArray()))
                            {
                                zip.createEntry(entry.getName(), in);
                            }
                        }
                    }
                    else
                    {
                        zip.createEntry(entry.getName() + "/", new java.io.ByteArrayInputStream(new byte[0]));
                    }
                }
            }
            zip.save("RARtoZIPoutput.zip");
        }
        catch (IOException ex) { System.out.println(ex);
        }

        System.out.println("RAR to ZIP converted successfully");
    }
}