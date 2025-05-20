//date: 2025-05-20T17:04:36Z
//url: https://api.github.com/gists/21c384e9be6d7c585ddf3b85c1fa111b
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.imaging.*;
import com.aspose.imaging.fileformats.tiff.enums.TiffPhotometrics;
import com.aspose.imaging.imageoptions.TiffOptions;
import com.aspose.imaging.fileformats.tiff.enums.TiffExpectedFormat;

public class Main {
    public static void main(String[] args) {
        // Set license
        com.aspose.imaging.License imagingLicense = new com.aspose.imaging.License();
        imagingLicense.setLicense("license.lic");

        // Load the image
        try (Image inputImage = Image.load("sample.jpg")) {
            // Ensure the image is a RasterImage
            RasterImage photo = (RasterImage) inputImage;

            // Cache data if not already done
            if (!photo.isCached()) {
                photo.cacheData();
            }

            // Adjust brightness
            photo.adjustBrightness(100);

            // Set TIFF save options
            TiffOptions tiffSettings = new TiffOptions(TiffExpectedFormat.Default);
            tiffSettings.setBitsPerSample(new int[] { 8, 8, 8 });
            tiffSettings.setPhotometric(TiffPhotometrics.Rgb);

            // Save the modified image as TIFF
            photo.save("result.tiff", tiffSettings);
        }

        // Notify completion
        System.out.println("Done");
    }
}
