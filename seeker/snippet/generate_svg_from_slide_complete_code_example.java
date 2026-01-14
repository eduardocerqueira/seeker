//date: 2026-01-14T17:18:43Z
//url: https://api.github.com/gists/53f191e8f5eff68149b5e4eee27f8b0f
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

import java.io.FileOutputStream;
import java.io.IOException;

public class PptxToSvgConverter {
    public static void main(String[] args) {
        // Set license - replace with your actual license file path
        try {
            License license = new License();
            license.setLicense("Aspose.Slides.Java.lic");
        } catch (Exception e) {
            System.out.println("License loading failed: " + e.getMessage());
            return;
        }

        // Load the presentation
        Presentation presentation;
        try {
            presentation = new Presentation("sample.pptx");
        } catch (Exception e) {
            System.out.println("Failed to load presentation: " + e.getMessage());
            return;
        }

        // Iterate through slides and export each as SVG
        for (int i = 0; i < presentation.getSlides().size(); i++) {
            ISlide slide = presentation.getSlides().get_Item(i);
            String outputPath = "output/slide_" + (i + 1) + ".svg";

            try (FileOutputStream outStream = new FileOutputStream(outputPath)) {
                // Optional: customize SVG options
                SvgOptions svgOptions = new SvgOptions();
                svgOptions.setEmbedFonts(true);
                slide.writeAsSvg(outStream, svgOptions);
                System.out.println("Exported: " + outputPath);
            } catch (IOException ex) {
                System.out.println("Error writing SVG for slide " + (i + 1) + ": " + ex.getMessage());
            }
        }

        // Cleanup
        presentation.dispose();
        System.out.println("All slides have been converted to SVG.");
    }
}