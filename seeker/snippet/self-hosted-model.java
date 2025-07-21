//date: 2025-07-21T17:10:26Z
//url: https://api.github.com/gists/d3bae01f9bc532c0f4739d05762b9799
//owner: https://api.github.com/users/aspose-words-gists

// For complete examples and data files, please go to https://github.com/aspose-words/Aspose.Words-for-Java.git.
public void selfHostedModel() throws Exception
{
    Document doc = new Document(getMyDir() + "Big document.docx");

    String apiKey = System.getenv("API_KEY");
    // Use OpenAI generative language models.
    AiModel model = new CustomAiModel().withApiKey(apiKey);

    Document translatedDoc = model.translate(doc, Language.RUSSIAN);
    translatedDoc.save(getArtifactsDir() + "AI.SelfHostedModel.docx");
}

/// <summary>
/// Custom self-hosted AI model.
/// </summary>
static class CustomAiModel extends OpenAiModel
{
    /// <summary>
    /// Gets custom URL of the model.
    /// </summary>
    protected /*override*/ String getUrl() { return "https://localhost/"; }

    /// <summary>
    /// Gets model name.
    /// </summary>
    protected /*override*/ String getName() { return "my-model-24b"; }
}