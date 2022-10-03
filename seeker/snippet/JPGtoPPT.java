//date: 2022-10-03T17:34:16Z
//url: https://api.github.com/gists/d98c0f7cb1659dabe88607497b477fd7
//owner: https://api.github.com/users/aspose-slides-gists

Presentation pres = new Presentation();
try {
    IPPImage image = pres.getImages().addImage(Files.readAllBytes(Paths.get("image.jpg")));
    pres.getSlides().get_Item(0).getShapes().addPictureFrame(ShapeType.Rectangle, 0, 0, 720, 540, image);
    pres.save("pres.pptx", SaveFormat.Pptx);
} catch(IOException e) {
} finally {
    if (pres != null) pres.dispose();
}