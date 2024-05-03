//date: 2024-05-03T16:57:53Z
//url: https://api.github.com/gists/f87b292613a42c5fd507b7f0a7a92388
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.zip.*;
public class Main
{
    public static void main(String[] args) throws Exception // Extract RAR file using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");

        // Load the input RAR file
        try (com.aspose.zip.RarArchive archive = new com.aspose.zip.RarArchive("input.rar"))
        {
            // Extract RAR files
            archive.extractToDirectory("outputDirectory");
        }
        catch (Exception ex)
        {
            System.out.println(ex.getMessage());
        }
        System.out.println("Done");
    }
}