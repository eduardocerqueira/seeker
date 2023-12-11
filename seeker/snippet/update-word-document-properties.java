//date: 2023-12-11T17:09:55Z
//url: https://api.github.com/gists/da3319b626b82e5db4ccfd4c6da6ba97
//owner: https://api.github.com/users/aspose-com-gists

 Document doc = new Document("SampleProps.docx");

CustomDocumentProperties custProps = doc.getCustomDocumentProperties();

if (custProps.get("Reviewed") != null) {            
   custProps.get("Reviewed By").setValue("Mart");
   custProps.get("Reviewed Date").setValue(new java.util.Date());
}

BuiltInDocumentProperties documentProperties = doc.getBuiltInDocumentProperties();

documentProperties.get("Pages").setValue(doc.getPageCount());
documentProperties.get("Comments").setValue("Document Comments");
documentProperties.get("Title").setValue("Document Title");

// Save the output file
doc.save("Output.docx");