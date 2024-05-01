//date: 2024-05-01T17:07:09Z
//url: https://api.github.com/gists/f47c06d5c3cc18a08585b167876dbd9f
//owner: https://api.github.com/users/groupdocs-com-kb

import com.groupdocs.signature.domain.enums.HorizontalAlignment;
import com.groupdocs.signature.domain.enums.VerticalAlignment;
import com.groupdocs.watermark.Watermarker;
import com.groupdocs.watermark.licenses.License;
import com.groupdocs.watermark.watermarks.Font;
import com.groupdocs.watermark.watermarks.SizingType;
import com.groupdocs.watermark.watermarks.TextWatermark;

public class AddTextWatermarktoRTFusingJava {

    public static void main(String[] args) {

        // Set License to avoid the limitations of Watermark library
        License license = new License();
        license.setLicense("GroupDocs.Watermark.lic");

        Watermarker watermarker = new Watermarker("input.rtf");

        Font font = new Font("Calibri", 8);
        TextWatermark watermark = new TextWatermark("Text watermark", font);
        watermark.setHorizontalAlignment(HorizontalAlignment.Right);
        watermark.setVerticalAlignment(VerticalAlignment.Top);
        watermark.setSizingType(SizingType.ScaleToParentDimensions);
        watermark.setScaleFactor(0.5);

        // Set rotation angle
        watermark.setRotateAngle(45);

        watermarker.add(watermark);
        watermarker.save("output.rtf");

        watermarker.close();
    }
}