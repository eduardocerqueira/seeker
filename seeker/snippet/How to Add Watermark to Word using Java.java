//date: 2022-10-17T17:26:21Z
//url: https://api.github.com/gists/88818a41802301df119281072e73c859
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.words.License;
import java.awt.Color;
import com.aspose.words.Document;
import com.aspose.words.TextWatermarkOptions;
import com.aspose.words.WatermarkLayout;

public class AsposeTest {

	public static void main(String[] args) throws Exception {//Main function to add watermark in Java

                // Instantiate the license
                License lic = new License(); 
                lic.setLicense("Aspose.Total.lic");

                // Create a Document class instance
                Document doc = new Document();

                // Instantiate the TextWatermarkOptions object
                TextWatermarkOptions options = new TextWatermarkOptions();
                
                // Set watermark properties
                options.setFontFamily("Calibri");
                options.setFontSize(42);
                options.setColor(Color.BLUE);
                options.setLayout(WatermarkLayout.DIAGONAL);
                options.isSemitrasparent(true);
                

                // Put the watermark text with options
                doc.getWatermark().setText("TRIAL VERSION WATERMARK", options);

                // Save the document
                doc.save("TextWatermark.docx");

                System.out.println("Done");
	}
}