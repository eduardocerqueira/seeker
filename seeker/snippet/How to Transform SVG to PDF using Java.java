//date: 2023-06-02T16:54:07Z
//url: https://api.github.com/gists/2bf8d91e89035a1b94998eb5f2ccd5c7
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.pdf.*;
public class Main {
    public static void main(String[] args) throws Exception // change SVG to PDF using java
    {
        // Set the license
        new License().setLicense("Aspose.Total.lic");

        SvgLoadOptions options = new SvgLoadOptions();

        // Load SVG file
        Document svgDoc = new Document("input.svg" , options);

        // Save the SVG as PDF
        svgDoc.save("output.pdf");

        System.out.println("Done");
    }
}