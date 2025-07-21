//date: 2025-07-21T17:10:26Z
//url: https://api.github.com/gists/d3bae01f9bc532c0f4739d05762b9799
//owner: https://api.github.com/users/aspose-words-gists

// For complete examples and data files, please go to https://github.com/aspose-words/Aspose.Words-for-Java.git.
Document doc = new Document(getMyDir() + "Tables.docx");

Row row = doc.getFirstSection().getBody().getTables().get(0).getFirstRow();
row.setHidden(true);

doc.save(getArtifactsDir() + "Table.HiddenRow.docx");

doc = new Document(getArtifactsDir() + "Table.HiddenRow.docx");

row = doc.getFirstSection().getBody().getTables().get(0).getFirstRow();
Assert.assertTrue(row.getHidden());

for (Cell cell : row.getCells())
{
    for (Paragraph para : cell.getParagraphs())
    {
        for (Run run : para.getRuns())
            Assert.assertTrue(run.getFont().getHidden());
    }
}