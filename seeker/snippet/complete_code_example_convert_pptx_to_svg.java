//date: 2026-01-14T17:10:08Z
//url: https://api.github.com/gists/1f1a5fe9d70ce1eaba02dfa6501c8cbc
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;

public class PptxToSvgConverter {
    public static void main(String[] args) {
        // Set license (replace with your license file path)
        try {
            License license = new License();
            license.setLicense("Aspose.Slides.Java.lic");
        } catch (Exception e) {
            System.out.println("License loading failed: " + e.getMessage());
        }

        // Input PPTX file
        String inputPath = "sample.pptx";

        // Output directory for SVG files
        String outputDir = "svg_output";
        new File(outputDir).mkdirs();

        try (Presentation pres = new Presentation(inputPath)) {
            int slideCount = pres.getSlides().size();

            for (int i = 0; i < slideCount; i++) {
                ISlide slide = pres.getSlides().get_Item(i);
                // Export slide to SVG string
                String svg = slide.getSvg();

                // Optional: customize SVG (example – change shape IDs)
                svg = svg.replaceAll("id=\"Shape", "id=\"CustomShape");

                // Write SVG to file
                Path outPath = Paths.get(outputDir, "slide_" + (i + 1) + ".svg");
                Files.write(outPath, svg.getBytes(StandardCharsets.UTF_8));
                System.out.println("Saved: " + outPath);
            }
        } catch (Exception ex) {
            System.err.println("Error during conversion: " + ex.getMessage());
        }
    }
}