//date: 2024-09-18T16:54:15Z
//url: https://api.github.com/gists/133d9815f8bd9b36601b1925952eeb4c
//owner: https://api.github.com/users/aspose-com-kb

import  com.aspose.words.*;

public class Main
{
    public static void main(String[] args) throws Exception // Adding shapes in Java
    {
        // Set the licenses
        new License().setLicense("License.lic");

        Document doc = new Document();
        DocumentBuilder builder = new DocumentBuilder(doc);

        //Inline shape
        Shape shape = builder.insertShape(ShapeType.LINE, 200, 200);
        shape.setRotation(35.0);

        //Free-floating shape
        shape = builder.insertShape
                ( ShapeType.ARROW,RelativeHorizontalPosition.PAGE,250,
                        RelativeVerticalPosition.PAGE,150,150,150,WrapType.INLINE);
        shape.setRotation(40.0);
        builder.writeln();
        OoxmlSaveOptions saveOptions = new OoxmlSaveOptions(SaveFormat.DOCX);

        // Save shapes as DML
        saveOptions.setCompliance(OoxmlCompliance.ISO_29500_2008_TRANSITIONAL);

        // Save the document
        doc.save("output.docx", saveOptions);

        System.out.println("Shapes added successfully");
    }
}