//date: 2025-12-15T17:00:14Z
//url: https://api.github.com/gists/05634c0c0a617b2e2696d0d5417dfac0
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

public class PptxToPdfConverter {

    public static void main(String[] args) {
        // Validate input arguments
        if (args.length != 2) {
            System.out.println("Usage: java PptxToPdfConverter <input-pptx> <output-pdf>");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        // Load Aspose.Slides license (optional but recommended)
        try {
            License license = new License();
            license.setLicense("Aspose.Slides.Java.lic");
        } catch (Exception e) {
            System.out.println("License not found or invalid. Continuing with evaluation mode.");
        }

        // Load the PPTX presentation
        Presentation presentation = null;
        try {
            presentation = new Presentation(inputPath);
        } catch (Exception e) {
            System.err.println("Error loading presentation: " + e.getMessage());
            return;
        }

        // Configure PDF export options
        PdfOptions pdfOptions = new PdfOptions();
        pdfOptions.setJpegQuality(100); // maximum image quality
        pdfOptions.setFontEmbeddingMode(FontEmbeddingMode.EMBED_ALL);
        pdfOptions.setIncludeSlideNotes(true);
        pdfOptions.setIncludeHyperlink(true);
        pdfOptions.setCompressImages(true);
        pdfOptions.setSimplifyShapes(false); // keep original shapes for fidelity

        // Save as PDF
        try {
            presentation.save(outputPath, SaveFormat.Pdf, pdfOptions);
            System.out.println("Conversion successful: " + outputPath);
        } catch (Exception e) {
            System.err.println("Error during PDF export: " + e.getMessage());
        } finally {
            if (presentation != null) {
                presentation.dispose();
            }
        }
    }
}