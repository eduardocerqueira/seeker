//date: 2023-06-08T17:05:04Z
//url: https://api.github.com/gists/7f4b44b640192cf21ab0e8e8309bd779
//owner: https://api.github.com/users/aspose-com-kb

public class Main {
    public static void main(String[] args) throws Exception // .ONE to DOC in Java
    {
        // Set the licenses
        new com.aspose.note.License().setLicense("Aspose.Total.lic");
        new com.aspose.words.License().setLicense("Aspose.Total.lic");

        // Convert .ONE to HTML
        com.aspose.note.Document doc = new com.aspose.note.Document("Aspose.One");
        doc.save("output.html", com.aspose.note.SaveFormat.Html);

        // Convert HTML to DOC
        com.aspose.words.Document document = new com.aspose.words.Document("output.html");

        // Save the DOC
        document.save("output.doc", com.aspose.words.SaveFormat.DOC);

        System.out.println("Done");
    }
}