//date: 2023-01-30T16:49:03Z
//url: https://api.github.com/gists/c32dec68b8d9d0f16b62f6fedb44e9cc
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.pdf.*;

public class Main {
    public static void main(String[] args) throws Exception {// Change order of pages

        // Load a license
        License lic = new License();
        lic.setLicense("Aspose.Total.lic");

        // Initialize document object
        Document srcDocument = new Document();

        // Add page
        for(int i = 1; i <= 10; i++) {
            TextFragment textFragment = new com.aspose.pdf.TextFragment("Text on page " + i);
            srcDocument.getPages().add().getParagraphs().add(textFragment);
        }

        var page = srcDocument.getPages().get_Item(2);
        srcDocument.getPages().add(page);
        srcDocument.getPages().delete(2);
        srcDocument.save("Output1.pdf");
        srcDocument.close();

        srcDocument = new Document("Output1.pdf");

        page = srcDocument.getPages().get_Item(3);
        srcDocument.getPages().insert(7,page);
        srcDocument.getPages().delete(3);
        srcDocument.save("result2.pdf");

        System.out.println("Done");
    }
}