//date: 2025-12-09T16:59:00Z
//url: https://api.github.com/gists/b34c8049b6e5b732055c6fb9ae6af7fd
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

import java.io.FileInputStream;
import java.io.InputStream;

public class PptxToPdfConverter {
    public static void main(String[] args) {
        // Validate arguments
        if (args.length != 2) {
            System.out.println("Usage: java PptxToPdfConverter <input.pptx> <output.pdf>");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        // Load Aspose.Slides license (optional but recommended)
        try (InputStream licStream = new FileInputStream("Aspose.Slides.lic")) {
            License license = new License();
            license.setLicense(licStream);
        } catch (Exception e) {
            System.out.println("License not found or invalid, proceeding in evaluation mode.");
        }

        // Load the PPTX presentation
        Presentation presentation = new Presentation(inputPath);

        // Set up PDF export options for high quality output
        PdfOptions pdfOptions = new PdfOptions();
        pdfOptions.setFontEmbeddingMode(FontEmbeddingMode.EMBED_ALL);
        pdfOptions.setJpegQuality(100);

        // Optional: configure vector rasterization for better image handling
        VectorRasterizationOptions vectorOpts = new VectorRasterizationOptions();
        vectorOpts.setPageSize(presentation.getSlideSize().getSize());
        pdfOptions.setVectorRasterizationOptions(vectorOpts);

        // Save as PDF
        presentation.save(outputPath, SaveFormat.Pdf, pdfOptions);

        System.out.println("Conversion completed successfully. PDF saved to " + outputPath);
    }
}