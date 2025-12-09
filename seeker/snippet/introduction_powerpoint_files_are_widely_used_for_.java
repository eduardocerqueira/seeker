//date: 2025-12-09T17:07:44Z
//url: https://api.github.com/gists/d259d1a3bf563ca823329403cf97c29e
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

public class PptxToPdfConverter {
    public static void main(String[] args) {
        // Path to the source PPTX and the output PDF
        String sourcePath = "input.pptx";
        String outputPath = "output.pdf";

        try {
            // Load Aspose.Slides license (optional but recommended for production)
            License license = new License();
            license.setLicense("Aspose.Slides.lic"); // place your license file in the project root

            // Load the presentation
            Presentation presentation = new Presentation(sourcePath);

            // Configure PDF export options
            PdfOptions pdfOptions = new PdfOptions();
            pdfOptions.setEmbedFullFonts(true);          // embed all fonts to preserve appearance
            pdfOptions.setJpegQuality(95);              // high quality images
            pdfOptions.setCompressImages(true);         // enable image compression
            pdfOptions.setPreserveTransition(true);     // keep slide transitions

            // Save the presentation as a high‑quality PDF
            presentation.save(outputPath, SaveFormat.Pdf, pdfOptions);

            System.out.println("Conversion successful. PDF saved at " + outputPath);
        } catch (Exception e) {
            System.err.println("Error during conversion: " + e.getMessage());
            e.printStackTrace();
        }
    }
}