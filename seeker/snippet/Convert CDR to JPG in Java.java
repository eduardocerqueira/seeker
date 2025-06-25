//date: 2025-06-25T16:56:59Z
//url: https://api.github.com/gists/4f6a8f46563d3a6bc2b9587325b3cdf0
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.imaging.*;
import com.aspose.imaging.fileformats.cdr.CdrImage;
import com.aspose.imaging.imageoptions.JpegOptions;
import com.aspose.imaging.imageoptions.VectorRasterizationOptions;

public class CdrToJpegConverter {
    public static void main(String[] args) {
        // Activate Aspose.Imaging features with a license file
        License license = new License();
        license.setLicense("license.lic");

        // Set the input and output file paths
        String sourceFile = "SampleCDRFile.cdr";
        String outputFile = "converted.jpg";

        // Load and convert the CDR file
        try (CdrImage cdr = (CdrImage) Image.load(sourceFile)) {
            // Configure rasterization options for vector rendering
            VectorRasterizationOptions rasterOptions = new VectorRasterizationOptions();
            rasterOptions.setBackgroundColor(Color.getWhite());
            rasterOptions.setTextRenderingHint(TextRenderingHint.SingleBitPerPixel);
            rasterOptions.setSmoothingMode(SmoothingMode.None);
            rasterOptions.setPageWidth(cdr.getWidth());
            rasterOptions.setPageHeight(cdr.getHeight());

            // Set up JPG export settings
            JpegOptions jpegOptions = new JpegOptions();
            jpegOptions.setVectorRasterizationOptions(rasterOptions);
            jpegOptions.setResolutionSettings(new ResolutionSetting(cdr.getWidth(), cdr.getHeight()));

            // Save the output image
            cdr.save(outputFile, jpegOptions);
            System.out.println("CDR file successfully exported to JPG.");
        } catch (Exception e) {
            System.err.println("Error during conversion: " + e.getMessage());
        }
    }
}
