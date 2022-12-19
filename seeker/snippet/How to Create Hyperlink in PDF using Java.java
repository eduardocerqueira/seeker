//date: 2022-12-19T16:39:36Z
//url: https://api.github.com/gists/31956524e2c0504e147585914b15622b
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.pdf.*;

public class Main {
    public static void main(String[] args) throws Exception {//Add hyperlink to PDF in Java

        // Load a license
        License lic = new License();
        lic.setLicense("Aspose.Total.lic");

        // Load the document
        Document document = new Document("AddHyperlink.pdf");

        // Get access to the first page for adding a hyperlink
        Page page = document.getPages().get_Item(1);

        // Instantiate a link annotation and set its properties
        LinkAnnotation link = new LinkAnnotation(page, new Rectangle(150, 150, 350, 350));
        Border border = new Border(link);
        border.setWidth(0);
        link.setBorder(border);
        link.setAction(new GoToURIAction("www.aspose.com"));

        // Add the annotation 
        page.getAnnotations().add(link);

        // Instantiate the free text annotation and set its properties
        FreeTextAnnotation textAnnotation = new FreeTextAnnotation(document.getPages().get_Item(1),
                new Rectangle(100, 100, 300, 300),
                new DefaultAppearance("TimesNewRoman", 10, Color.getBlue().toRgb()));
        textAnnotation.setContents("Link to Aspose website");

        // Set the border
        textAnnotation.setBorder(border);

        // Add the text annotation to the page at the same location where link annotation is added
        page.getAnnotations().add(textAnnotation);

        // Save the updated PDF document
        document.save("Output.pdf");

        System.out.println("Done");
    }
}