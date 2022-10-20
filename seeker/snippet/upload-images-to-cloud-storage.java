//date: 2022-10-20T17:31:53Z
//url: https://api.github.com/gists/b5b3373fc27191a7e25fe14fd262a530
//owner: https://api.github.com/users/blog-aspose-cloud

// Get ClientID and ClientSecret from https: "**********"
String clientId = "7ef10407-c1b7-43bd-9603-5ea9c6db83cd";
String clientSecret = "**********"

// create Imaging object
ImagingApi imageApi = "**********"

File directory = new File("/Users/");
//Get all the files from the folder
File[] allFiles = directory.listFiles();
if (allFiles == null || allFiles.length == 0) {
    throw new RuntimeException("No files present in the directory: " + directory.getAbsolutePath());
}
			 
//Set the required image extensions here.
List<String> supportedImageExtensions = Arrays.asList("jpg", "png", "gif", "webp");
			 
int counter =0;
//Filter out only image files
List<File> acceptedImages = new ArrayList<>();
for (File file : allFiles) {
    //Parse the file extension
    String fileExtension = file.getName().substring(file.getName().lastIndexOf(".") + 1);
    //Check if the extension is listed in the supportedImageExtensions
    if (supportedImageExtensions.stream().anyMatch(fileExtension::equalsIgnoreCase)) {
        //Add the image to the filtered list
        acceptedImages.add(file);
			    
    // load first PowerPoint presentation
    byte[] bytes = Files.readAllBytes(file.toPath());
	
    // create file upload request
    UploadFileRequest request = new UploadFileRequest(acceptedImages.get(counter).getName(),bytes,null);
    // upload image file to cloud storage
    imageApi.uploadFile(request);
    // increase file counter
    counter+=1;
    }
}request);
    // increase file counter
    counter+=1;
    }
}