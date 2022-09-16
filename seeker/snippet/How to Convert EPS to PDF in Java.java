//date: 2022-09-16T17:21:12Z
//url: https://api.github.com/gists/978650e85e034700b80b7ef6cc4f010c
//owner: https://api.github.com/users/aspose-com-kb

import java.io.FileOutputStream;

public class AsposeTest {

	public static void main(String[] args) throws Exception {//Main function to convert EPS to PDF
		
		// Instantiate the license
		com.aspose.page.License licPage = new com.aspose.page.License(); 
		licPage.setLicense("Aspose.Total.lic");
		
		 // Initialize PDF stream
		FileOutputStream pdfStream = new FileOutputStream("EPStoPDF.pdf");
		
		// Initialize PostScript stream
		java.io.FileInputStream psStream = new java.io.FileInputStream("circle.eps");
		
		// Create PsDocument class object
		com.aspose.eps.PsDocument document = new com.aspose.eps.PsDocument(psStream);
		
		// Initialize PdfSaveOptions object.
		com.aspose.eps.device.PdfSaveOptions options = new com.aspose.eps.device.PdfSaveOptions(true);
		
		// Create a PdfDevice
		com.aspose.eps.device.PdfDevice device = new com.aspose.eps.device.PdfDevice(pdfStream);
		
		try {
		    document.save(device, options);
		} finally {
		    psStream.close();
		    pdfStream.close();
		}

        System.out.println("Done");
	}
}