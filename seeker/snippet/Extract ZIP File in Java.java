//date: 2024-05-03T16:56:40Z
//url: https://api.github.com/gists/db48833990b1e881cf16588f46875a0e
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.zip.*;
public class Main
{
    public static void main(String[] args) throws Exception // Extract ZIP file using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");
        // Load the input ZIP file
        try (com.aspose.zip.Archive archive = new com.aspose.zip.Archive("input.zip"))
        {
            // Extract ZIP files
            archive.extractToDirectory("outputDirectory");
        }
        catch (Exception ex)
        {
            System.out.println(ex.getMessage());
        }
        System.out.println("Done");
    }
}