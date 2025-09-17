//date: 2025-09-17T17:08:49Z
//url: https://api.github.com/gists/1b601d92d078262bef92ef59ef2c0cef
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.font.*;  // Aspose.Font classes
import java.io.FileOutputStream;
import java.io.IOException;

public class TtfToWoff {
    public static void main(String[] args) {
        try {
            // Load Aspose.Font license
            License lic = new License();
            lic.setLicense("License.lic");

            // Open TTF font
            String fontPath = "Geneva.ttf";
            FontDefinition fontDefinition = new FontDefinition(
                    FontType.TTF,
                    new FontFileDefinition(new FileSystemStreamSource(fontPath))
            );
            Font font = Font.open(fontDefinition);

            // WOFF output settings
            String outPath = "Geneva.woff";
            FileOutputStream outStream = new FileOutputStream(outPath);

            // Convert TTF to WOFF
            font.saveToFormat(outStream, FontSavingFormats.WOFF);

            outStream.close();
            System.out.println("TTF successfully converted to WOFF!");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
