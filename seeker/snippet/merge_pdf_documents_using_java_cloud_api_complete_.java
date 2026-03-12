//date: 2026-03-12T17:28:11Z
//url: https://api.github.com/gists/f3955b6a13d13a84488607cf23fd77a3
//owner: https://api.github.com/users/conholdate-cloud-gists

import com.conholdate.total.api.PdfApi;
import com.conholdate.total.model.MergePdfRequest;
import java.nio.file.Files;
import java.nio.file.Paths;

public class PdfMergeDemo {
    public static void main(String[] args) {
        // Replace with your actual credentials
        String clientId = "YOUR_CLIENT_ID";
        String clientSecret = "**********"

        try {
            // Initialize the API client
            PdfApi pdfApi = "**********"

            // Upload source PDFs
            pdfApi.uploadFile("doc1.pdf", Files.readAllBytes(Paths.get("doc1.pdf")));
            pdfApi.uploadFile("doc2.pdf", Files.readAllBytes(Paths.get("doc2.pdf")));

            // Prepare merge request
            MergePdfRequest mergeRequest = new MergePdfRequest()
                    .addInputFile("doc1.pdf")
                    .addInputFile("doc2.pdf")
                    .setOutputFile("merged_result.pdf");

            // Perform merge
            pdfApi.mergePdf(mergeRequest);

            // Download merged PDF
            byte[] mergedData = pdfApi.downloadFile("merged_result.pdf");
            Files.write(Paths.get("merged_result.pdf"), mergedData);

            System.out.println("PDF files merged successfully.");
        } catch (Exception e) {
            System.err.println("Error during PDF merge: " + e.getMessage());
            e.printStackTrace();
        }
    }
}tStackTrace();
        }
    }
}