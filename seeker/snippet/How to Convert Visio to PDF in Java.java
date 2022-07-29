//date: 2022-07-29T17:06:11Z
//url: https://api.github.com/gists/4c3ba38d282966d6a529f0cc3a41aa61
//owner: https://api.github.com/users/aspose-com-kb

public class AsposeTest {

	public static void main(String[] args) throws Exception {//Main function to convert VSD to PDF 

		// Instantiate the license
		com.aspose.diagram.License license = new com.aspose.diagram.License(); 
		license.setLicense("Aspose.Total.lic");

		// Load Visio diagram
		com.aspose.diagram.Diagram diagram = new com.aspose.diagram.Diagram("Sample.vsd");

		// Declare PdfSaveOptions object
		com.aspose.diagram.PdfSaveOptions saveOptions = new com.aspose.diagram.PdfSaveOptions();

		// Number of pages to render
		saveOptions.setPageCount(2);

		// Set first page index
		saveOptions.setPageIndex(1);

		// Save Visio diagram to PDF
		diagram.save("PDF_out.pdf", saveOptions);

		System.out.println("Done");
	}
}