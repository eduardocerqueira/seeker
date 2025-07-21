//date: 2025-07-21T17:10:26Z
//url: https://api.github.com/gists/d3bae01f9bc532c0f4739d05762b9799
//owner: https://api.github.com/users/aspose-words-gists

// For complete examples and data files, please go to https://github.com/aspose-words/Aspose.Words-for-Java.git.
@Test
public void matchEndNode() throws Exception
{
    Document doc = new Document();
    DocumentBuilder builder = new DocumentBuilder(doc);

    builder.writeln("1");
    builder.writeln("2");
    builder.writeln("3");

    ReplacingCallback replacingCallback = new ReplacingCallback();
    FindReplaceOptions opts = new FindReplaceOptions();
    opts.setReplacingCallback(replacingCallback);

    doc.getRange().replace(Pattern.compile("1[\\s\\S]*3"), "X", opts);
    Assert.assertEquals("1", replacingCallback.getStartNodeText());
    Assert.assertEquals("3", replacingCallback.getEndNodeText());
}

/// <summary>
/// The replacing callback.
/// </summary>
private static class ReplacingCallback implements IReplacingCallback
{
    public int replacing(ReplacingArgs e)
    {
        setStartNodeText(e.getMatchNode().getText().trim());
        setEndNodeText(e.getMatchEndNode().getText().trim());

        return ReplaceAction.REPLACE;
    }

    private String mStartNodeText;
    String getStartNodeText() { return mStartNodeText; }; private void setStartNodeText(String value) { mStartNodeText = value; };

    private String mEndNodeText;
    String getEndNodeText() { return mEndNodeText; }; private void setEndNodeText(String value) { mEndNodeText = value; };
}