//date: 2025-12-15T17:03:13Z
//url: https://api.github.com/gists/dccb608e264599507cfa03266edb0fe0
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.slides.*;
import com.aspose.slides.export.*;
import java.io.File;
import java.nio.file.*;

public class PptxToPdfConverter {

    public static void main(String[] args) throws Exception {
        // Path to a single PPTX file or a folder for batch conversion
        String inputPath = "inputFolder"; // change to your folder or file path
        String outputFolder = "outputPdf";

        // Ensure output directory exists
        Files.createDirectories(Paths.get(outputFolder));

        File input = new File(inputPath);
        if (input.isDirectory()) {
            // Batch conversion
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(input.toPath(), "*.pptx")) {
                for (Path pptxPath : stream) {
                    convertFile(pptxPath.toString(), outputFolder);
                }
            }
        } else {
            // Single file conversion
            convertFile(inputPath, outputFolder);
        }
    }

    private static void convertFile(String pptxFile, String outputFolder) throws Exception {
        // Load presentation
        try (Presentation pres = new Presentation(pptxFile)) {
            // Configure PDF options
            PdfOptions pdfOptions = new PdfOptions();
            pdfOptions.setEmbedFullFonts(true);
            // Set font folder if you have custom fonts
            pdfOptions.setFontFolder("C:/MyFonts", true);
            pdfOptions.setCompressionLevel(CompressionLevel.Maximum);
            pdfOptions.setImageQuality(90);
            // Optional: PDF/A compliance for archiving
            pdfOptions.setCompliance(PdfCompliance.PdfA1b);

            // Build output file name
            String pdfFileName = Paths.get(outputFolder,
                    Paths.get(pptxFile).getFileName().toString().replaceAll("\\.pptx$", ".pdf"))
                    .toString();

            // Save as PDF
            pres.save(pdfFileName, SaveFormat.Pdf, pdfOptions);
            System.out.println("Converted: " + pptxFile + " -> " + pdfFileName);
        }
    }
}