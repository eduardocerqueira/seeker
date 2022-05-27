//date: 2022-05-27T17:12:18Z
//url: https://api.github.com/gists/1acb32c235bf162abb5b9627c43fd273
//owner: https://api.github.com/users/aspose-imaging-cloud-examples

import com.aspose.imaging.cloud.sdk.model.requests.CreateGrayscaledImageRequest;
import com.aspose.imaging.cloud.sdk.model.requests.GrayscaleImageRequest;

import java.nio.file.Files;
import java.nio.file.Paths;

string ImageFileName = "example_image.tga";
string ImagesFolder = "ExampleImages";
string CloudFolder = "CloudImages";
string OutputFolder = "Output";

// Get ClientId and ClientSecret from https://dashboard.aspose.cloud/ 
// or use on-premise version (https://docs.aspose.cloud/imaging/getting-started/how-to-run-docker-container/)
ImagingApi api = new ImagingApi(argumentValues.ClientSecret, argumentValues.ClientId, "https://api.aspose.cloud");

/**
* Grayscale an image from cloud storage.
*
* @throws Exception
*/
public void grayscaleImageFromStorage() throws Exception {

    // Upload image to cloud storage
    byte[] inputImage = Files.readAllBytes(Paths.get(ImagesFolder, ImageFileName));
    UploadFileRequest request = new UploadFileRequest(Paths.get(CloudFolder, ImageFileName).toString(), image, null);
    FilesUploadResult response = api.uploadFile(request);
    if(response.getErrors() != null && response.getErrors().size() > 0)
        throw new Exception("Uploading errors count: " + response.getErrors().size());

    String folder = CloudFolder; // Input file is saved at the desired folder in the storage
    String storage = null; // We are using default Cloud Storage

    GrayscaleImageRequest request = new GrayscaleImageRequest(ImageFileName, folder, storage);
    byte[] updatedImage = api.grayscaleImage(request);

    // Save the image file to output folder
    String updatedImageName = "changed_" + ImageFileName;
    Path path = Paths.get(OutputFolder, updatedImageName).toAbsolutePath();
    Files.write(path, updatedImage);
}

/**
* Grayscale an image. Image data is passed in a request stream.
*
* @throws Exception
*/
public void createGrayscaledImageFromRequest() throws Exception {

    byte[] inputStream = Files.readAllBytes(Paths.get(ImagesFolder, ImageFileName));

    String outPath = null; // Path to updated file (if this is empty, response contains streamed image)
    String storage = null; //  We are using default Cloud Storage

    CreateGrayscaledImageRequest request = new CreateGrayscaledImageRequest(inputStream, outPath, storage);
    byte[] updatedImage = api.createGrayscaledImage(request);  
    
    // Save the image file to output folder
    String updatedImageName = "changed_" + ImageFileName;
    Path path = Paths.get(OutputFolder, updatedImageName).toAbsolutePath();
    Files.write(path, updatedImage);
}
