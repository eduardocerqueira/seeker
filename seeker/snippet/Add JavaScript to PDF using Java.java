//date: 2026-03-17T17:37:23Z
//url: https://api.github.com/gists/1eaecefcb5989d11610bc03c19ab49cb
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.pdf.Document;
import com.aspose.pdf.JavascriptAction;

public class AddJsToPdf
{
    public static void main(String[] args) throws Exception
    {
        // Load Aspose license
        com.aspose.pdf.License license = new com.aspose.pdf.License();
        license.setLicense("license.lic");
        try
        {
            // Load an existing PDF
            Document document = new Document("input.pdf");

            // Add JavaScript that runs when the PDF is opened
            JavascriptAction js = new JavascriptAction(
                    "app.alert('Welcome! This PDF contains JavaScript.');"
            );

            document.setOpenAction(js);

            // Add JavaScript that runs before the PDF is closed
            document.getActions().setBeforeClosing(
                    new JavascriptAction(
                            "app.alert('Thank you for viewing this PDF!');"
                            // "app.launchURL(\"https://example.com\");"
                    )
            );

            // Save the updated PDF
            document.save("output-with-javascript.pdf");
            System.out.println("Done");

        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
