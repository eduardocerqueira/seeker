//date: 2023-11-20T16:56:49Z
//url: https://api.github.com/gists/df2c7888484719e37e622df249b8b2d1
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.cells.*;
public class Main
{
    public static void main(String[] args) throws Exception // Wrap text using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");

        // Create  a workbook and access a sheet
        Workbook wb = new Workbook();
        Worksheet ws = wb.getWorksheets().get(0);

        // Put text in different cells
        Cell c1 = ws.getCells().get("C1");
        c1.putValue("We will not wrap this text");
        Cell c5 = ws.getCells().get("C5");
        c5.putValue("We will wrap this text");

        // Set the wrap text style
        Style style = c5.getStyle();
        style.setTextWrapped(true);
        c5.setStyle(style);

        // Autofit rows
        ws.autoFitRows();

        // Save the file
        wb.save("output.xlsx", SaveFormat.XLSX);

        System.out.println("Done");
    }
}
