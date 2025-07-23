//date: 2025-07-23T17:04:17Z
//url: https://api.github.com/gists/d4391c91cbf14d0926f8230c6d8bd6ce
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.imaging.Image;
import com.aspose.imaging.RotateFlipType;
import com.aspose.imaging.fileformats.dng.DngImage;
import com.aspose.imaging.fileformats.jpeg.JpegCompressionColorMode;
import com.aspose.imaging.imageoptions.JpegOptions;

public class DngToJpgConverter {
    public static void main(String[] args) {
        // Load the DNG image
        try (DngImage image = (DngImage) Image.load("sample.dng")) {

            // Configure JPEG output options
            JpegOptions options = new JpegOptions();
            options.setQuality(50);
            options.setColorType(JpegCompressionColorMode.YCbCr); // Set color model

            // Optional image modifications
            image.rotateFlip(RotateFlipType.Rotate90FlipNone); // Rotate 90°
            image.resize(400, 300); // Resize image

            // Save the image as JPEG
            image.save("result.jpg", options);

            System.out.println("DNG to JPG conversion done successfully");
        }
    }
}
