//date: 2025-06-18T17:08:29Z
//url: https://api.github.com/gists/034aeb1a38aa63691f1abbf29dcebf78
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.imaging.Image;
import com.aspose.imaging.License;
import com.aspose.imaging.Color;
import com.aspose.imaging.SmoothingMode;
import com.aspose.imaging.TextRenderingHint;
import com.aspose.imaging.ResolutionSetting;
import com.aspose.imaging.fileformats.cdr.CdrImage;
import com.aspose.imaging.fileformats.png.PngColorType;
import com.aspose.imaging.imageoptions.PngOptions;
import com.aspose.imaging.imageoptions.VectorRasterizationOptions;

public class Main {
    public static void main(String[] args) {
        // Apply the license to enable full Aspose.Imaging functionality
        License imagingLicense = new License();
        imagingLicense.setLicense("license.lic");

        // Define path to the input CDR file
        String inputCdrPath = "SampleCDRFile.cdr";

        // Use try-with-resources to ensure proper resource cleanup
        try (CdrImage vectorImage = (CdrImage) Image.load(inputCdrPath)) {

            // Set up rasterization and output image settings
            VectorRasterizationOptions rasterSettings = new VectorRasterizationOptions();
            rasterSettings.setBackgroundColor(Color.getWhite());
            rasterSettings.setTextRenderingHint(TextRenderingHint.SingleBitPerPixel);
            rasterSettings.setSmoothingMode(SmoothingMode.None);

            PngOptions exportSettings = new PngOptions();
            exportSettings.setVectorRasterizationOptions(rasterSettings);
            exportSettings.setResolutionSettings(
                    new ResolutionSetting(vectorImage.getWidth(), vectorImage.getHeight()));
            exportSettings.setColorType(PngColorType.TruecolorWithAlpha);

            // Export to PNG image format
            String destinationPngPath = "result.png";
            vectorImage.save(destinationPngPath, exportSettings);

            // Notify user about the completion of the conversion
            System.out.println("Conversion from CDR to PNG completed successfully.");
        }
    }
}
