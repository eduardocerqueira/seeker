//date: 2026-01-14T17:13:02Z
//url: https://api.github.com/gists/222af7e05390ad84f96263f06657a865
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

import java.io.FileOutputStream;

public class PresentationToSvg {
    public static void main(String[] args) {
        // Path to the source PPTX file
        String sourcePath = "input.pptx";

        // Load the presentation inside a try‑with‑resources block
        try (Presentation pres = new Presentation(sourcePath)) {
            // Loop through all slides
            for (int i = 0; i < pres.getSlides().size(); i++) {
                ISlide slide = pres.getSlides().get_Item(i);
                String outFile = "slide_" + (i + 1) + ".svg";

                // Export each slide as SVG
                try (FileOutputStream outStream = new FileOutputStream(outFile)) {
                    slide.writeAsSvg(outStream);
                }
            }
            System.out.println("All slides have been exported to SVG successfully.");
        } catch (Exception e) {
            System.err.println("Error during SVG export: " + e.getMessage());
            e.printStackTrace();
        }
    }
}