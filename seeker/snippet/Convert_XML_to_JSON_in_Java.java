//date: 2023-03-14T17:05:49Z
//url: https://api.github.com/gists/256da93111cac94d68c4bd8ef8dff1fa
//owner: https://api.github.com/users/groupdocs-cloud-gists

// Prepare convert settings
ConvertSettings settings = new ConvertSettings();
settings.setFilePath("java-testing/input-sample-file.xml");
settings.setFormat("json");

settings.setOutputPath("java-testing/output-sample-file.json");

// convert to specified format
List<StoredConvertedResult> response = apiInstance.convertDocument(new ConvertDocumentRequest(settings));
System.out.println("Document converted successfully: " + response);