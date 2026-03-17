//date: 2026-03-17T17:39:12Z
//url: https://api.github.com/gists/320e7de4983c27b394e853ed4aea5ef8
//owner: https://api.github.com/users/mustafabutt-dev

import com.aspose.threed.cloud.api.ConversionApi;
import com.aspose.threed.cloud.model.ConvertHtmlToPptxRequest;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class HtmlToPptxDemo {
    public static void main(String[] args) {
        // Replace with your actual credentials
        String clientId = "YOUR_CLIENT_ID";
        String clientSecret = "**********"

        ConversionApi api = "**********"

        try {
            // Load HTML file
            File htmlFile = new File("sample.html");
            if (!htmlFile.exists()) {
                System.err.println("HTML source file not found.");
                return;
            }

            // Create conversion request
            ConvertHtmlToPptxRequest request = new ConvertHtmlToPptxRequest(htmlFile);

            // Perform conversion
            byte[] pptxBytes = api.convertHtmlToPptx(request);

            // Save PPTX to disk
            try (FileOutputStream fos = new FileOutputStream("output.pptx")) {
                fos.write(pptxBytes);
                System.out.println("Conversion successful. File saved as output.pptx");
            }
        } catch (IOException e) {
            System.err.println("IO error during conversion: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Conversion failed: " + e.getMessage());
        }
    }
} " + e.getMessage());
        }
    }
}