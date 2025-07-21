//date: 2025-07-21T17:10:44Z
//url: https://api.github.com/gists/80cfd78d9a0887277d6b4a391c73b04f
//owner: https://api.github.com/users/aspose-words-gists

// For complete examples and data files, please go to https://github.com/aspose-words/Aspose.Words-for-Java.git.
Document doc = new Document(getMyDir() + "Rendering.docx");

ImageSaveOptions options = new ImageSaveOptions(SaveFormat.JPEG);
// Set up a grid layout with:
// - 3 columns per row.
// - 10pts spacing between pages (horizontal and vertical).
options.setPageLayout(MultiPageLayout.grid(3, 10f, 10f));

// Alternative layouts:
// options.PageLayout = MultiPageLayout.Horizontal(10);
// options.PageLayout = MultiPageLayout.Vertical(10);

// Customize the background and border.
options.getPageLayout().setBackColor(Color.lightGray);
options.getPageLayout().setBorderColor(Color.BLUE);
options.getPageLayout().setBorderWidth(2f);

doc.save(getArtifactsDir() + "ImageSaveOptions.GridLayout.jpg", options);