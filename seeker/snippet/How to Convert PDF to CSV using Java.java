//date: 2022-02-22T16:55:25Z
//url: https://api.github.com/gists/9e0dc037e67bc71af23897336383b4f5
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.pdf.Document;
import com.aspose.pdf.ExcelSaveOptions;
import com.aspose.pdf.License;

public class ConvertPdfToCsvUsingJava {
    public static void main(String[] args) throws Exception { // main method to convert a PDF document to CSV file format
            // Instantiate the license to avoid trial limitations while converting the PDF to comma separated CSV file
            License asposePdfLicenseCSV = new License();
            asposePdfLicenseCSV.setLicense("Aspose.pdf.lic");
            
            // Load PDF document for converting it to comma seaparated value file
            Document convertPDFDocumentToCSV = new Document("input.pdf");

            // Initialize ExcelSaveOptions class object to set the format of the output file
            ExcelSaveOptions csvSave = new ExcelSaveOptions();
            csvSave.setFormat(ExcelSaveOptions.ExcelFormat.CSV);
            
            // Save the converted output file in CSV format
            convertPDFDocumentToCSV.save("ConvertPDFToCSV.csv", csvSave);

            System.out.println("Done");
    }
}