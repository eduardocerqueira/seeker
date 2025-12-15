//date: 2025-12-15T17:11:51Z
//url: https://api.github.com/gists/677a603ee015f54966dbce6956c399c3
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

public class PptxToPdfConverter {
    public static void main(String[] args) {
        // Validate arguments
        if (args.length != 2) {
            System.out.println("Usage: java PptxToPdfConverter <input.pptx> <output.pdf>");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        // Load Aspose.Slides license
        try {
            License license = new License();
            license.setLicense("Aspose.Slides.lic");
        } catch (Exception e) {
            System.err.println("License loading failed: " + e.getMessage());
            // Continue without license (evaluation mode) or exit based on requirements
        }

        // Register custom fonts folder (optional)
        Fonts.setFontFolder("C:/MyFonts", true);

        // Create Presentation object
        try (Presentation pres = new Presentation(inputPath)) {
            // Enable memory optimization for large files or batch processing
            pres.setMemoryOptimizationEnabled(true);

            // Configure PDF options
            PdfOptions pdfOptions = new PdfOptions();
            pdfOptions.setCompliance(PdfCompliance.PdfA1b);
            pdfOptions.setCompressionLevel(CompressionLevel.Maximum);
            // Save as PDF
            pres.save(outputPath, SaveFormat.Pdf, pdfOptions);
            System.out.println("Conversion successful: " + outputPath);
        } catch (Exception ex) {
            System.err.println("Error during conversion: " + ex.getMessage());
        }
    }
}