//date: 2025-12-09T17:12:36Z
//url: https://api.github.com/gists/a5d7de06cb0b03b8e8686d0da44ac906
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;

public class PptxToPdfConverter {
    public static void main(String[] args) {
        // Validate input arguments
        if (args.length != 2) {
            System.out.println("Usage: java PptxToPdfConverter <input.pptx> <output.pdf>");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        // Initialize presentation
        try (Presentation pres = new Presentation(inputPath)) {

            // Configure PDF export options
            PdfOptions pdfOptions = new PdfOptions();
            pdfOptions.setPreserveHyperlink(true);          // Keep hyperlinks active
            pdfOptions.setEmbedFullFonts(true);            // Embed all fonts
            pdfOptions.setSaveMetafilesAsPng(true);        // Render metafiles as PNG for better compatibility
            pdfOptions.setNotesCommentsLayoutingNotesPosition(NotesCommentsLayoutingOptions.NotesPositions.BottomFull); // Include slide notes

            // Save presentation as PDF
            pres.save(outputPath, SaveFormat.Pdf, pdfOptions);
            System.out.println("Conversion completed successfully: " + outputPath);
        } catch (Exception e) {
            System.err.println("Error during conversion: " + e.getMessage());
            e.printStackTrace();
        }
    }
}