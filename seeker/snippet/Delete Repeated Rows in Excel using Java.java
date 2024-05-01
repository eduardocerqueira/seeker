//date: 2024-05-01T17:01:27Z
//url: https://api.github.com/gists/f8dacc00088ca682323f8615efe340ff
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.cells.*;

public class Main
{
    public static void main(String[] args) throws Exception // Remove duplicate rows in Excel using Java
    {
        // Set the licenses
        new License().setLicense("License.lic");
        Workbook book = new Workbook("source.xlsx");

        // Remove duplicates from the entire sheet
        book.getWorksheets().get(2).getCells().removeDuplicates();

        // Remove duplicates
        book.getWorksheets().get(0).getCells().removeDuplicates(2,7,5,8);

        // Define reference columns
        int[] cols = { 1,2,4 };
        book.getWorksheets().get(0).getCells().removeDuplicates(1, 1, 6, 3,true,cols);

        //Save the result
        book.save("EliminateDuplicates.xlsx");

        System.out.println("Done");
    }
}