//date: 2024-04-26T16:35:41Z
//url: https://api.github.com/gists/92b10fb3135dba0f3d4bbec032d45a20
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.zip.*;

public class Main
{
    public static void main(String[] args) throws Exception // Password protect a ZIP file using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");

        // Create Archive class object using ArchiveEntrySettings
        var archive = new com.aspose.zip.Archive(new com.aspose.zip.ArchiveEntrySettings(null,
                new com.aspose.zip.AesEncryptionSettings("p@s$", com.aspose.zip.EncryptionMethod.AES256)));

        // Create entry for the ZIP file
        archive.createEntry("input.png","sample.png");

        // Save output protected ZIP file
        archive.save("PasswordAES256.zip");

        System.out.println("ZIP file password protected successfully");
    }
}