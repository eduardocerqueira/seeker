//date: 2022-02-02T17:11:03Z
//url: https://api.github.com/gists/c169e285c846ca3495bb6a5e9ae7e538
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.cells.Encoding;
import com.aspose.cells.HtmlSaveOptions;
import com.aspose.cells.License;
import com.aspose.cells.Workbook;

public class ConvertExcelToHtmlInJava {

	public static void main(String[] args) throws Exception {
		
		// Before converting Excel to HTML, load license to avoid watremark in the output HTML file
		License licenseForExcelToHtml = new License(); 
		licenseForExcelToHtml.setLicense("Aspose.Cells.lic");

		// Load the source input file to be converted to HTML 
		Workbook workbookToHtml = new Workbook("Sample.xlsx");

		// Create and initialize the save options for the HTML
		HtmlSaveOptions htmlSaveOptionsForExcel = new HtmlSaveOptions();
		
		// Set the encoding in the output HTML
		htmlSaveOptionsForExcel.setEncoding(Encoding.getUTF8());
		
		// Set the image format in the output HTML
		htmlSaveOptionsForExcel.setExportImagesAsBase64(true);
		
		// Set flag to display grid lines in the output HTML
		htmlSaveOptionsForExcel.setExportGridLines(true);
		
		// Set the columns width according to the contents for better visibility in output HTML
		workbookToHtml.getWorksheets().get(0).autoFitColumns();
		
		// Save the workbook as HTML using the above settings
		workbookToHtml.save("OutputHtmlFile.html", htmlSaveOptionsForExcel);
	}
}