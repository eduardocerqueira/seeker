//date: 2023-07-11T17:05:21Z
//url: https://api.github.com/gists/8a19b71afe5eccbf910589d285f273eb
//owner: https://api.github.com/users/groupdocs-com-kb

import com.groupdocs.signature.Signature;
import com.groupdocs.signature.domain.enums.DocumentType;
import com.groupdocs.signature.licensing.License;
import com.groupdocs.signature.options.sign.DigitalSignOptions;

public class AddDigitalSignaturetoDOCXusingJava {
    public static void main(String[] args) throws Exception {

        // Set License to avoid the limitations of Signature library
        License license = new License();
        license.setLicense("GroupDocs.Signature.lic");

        // load the source DOCX file
        Signature signature = new Signature("input.docx");

        // Create a digital signature option
        DigitalSignOptions options = new DigitalSignOptions("certificate.pfx");

        // Set the properties for signature appearance in DOCX
        options.setDocumentType(DocumentType.WordProcessing);
        // certificate password
        options.setPassword("password");
        // digital certificate details
        options.setReason("Approved");
        options.setContact("John Smith");
        options.setLocation("New York");

        options.setVisible(true);
        options.setImageFilePath( "signature.jpg");
        options.setLeft(100);
        options.setTop(100);
        options.setWidth (200);
        options.setHeight(50);

        // Save output DOCX to disk
        signature.sign("output.docx", options);
    }
}
