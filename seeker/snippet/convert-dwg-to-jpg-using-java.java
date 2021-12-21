//date: 2021-12-21T16:58:42Z
//url: https://api.github.com/gists/2241f420b14ea13058f4b968a98ff7ba
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.cad.License;
import com.aspose.cad.Color;
import com.aspose.cad.Image;
import com.aspose.cad.ImageOptionsBase;
import com.aspose.cad.fileformats.cad.CadDrawTypeMode;
import com.aspose.cad.imageoptions.CadRasterizationOptions;
import com.aspose.cad.imageoptions.JpegOptions;


public class ConvertDWGtoJPG {
    
    public static void main(String[] args) throws Exception { // main method to convert DWG to JPG image using Java

            // Set Aspose.CAD license before converting DWG to PNG Image
            License CADLicenseJava = new License();
            CADLicenseJava.setLicense("CADJavaLicense.lic");

            // Load the DWG to export to JPG
            Image image = Image.load("Test.dwg");
            
            // Create an instance of CadRasterizationOptions
            CadRasterizationOptions rasterizationOptions = new CadRasterizationOptions();

            // Set page width & height
            rasterizationOptions.setPageWidth(1200);
            rasterizationOptions.setPageHeight(1200);

            //Set background color and object colors
            rasterizationOptions.setBackgroundColor(Color.getWhite());
            rasterizationOptions.setDrawType(CadDrawTypeMode.UseObjectColor);

            // Create an instance of JpegOption for the converted Jpeg image
            ImageOptionsBase options = new JpegOptions();

            // Set rasterization options for exporting to JPEG
            options.setVectorRasterizationOptions(rasterizationOptions);

            // Save DWG to JPEG image
            image.save("Exported_image_out.jpeg", options);
    }
}
