//date: 2023-11-21T17:02:32Z
//url: https://api.github.com/gists/0ed9c2710c12fbbc1fe2bb4dfffe6629
//owner: https://api.github.com/users/halitkalayci

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ImageController {

    private final CloudinaryUploader cloudinaryUploader;

    public ImageController(CloudinaryUploader cloudinaryUploader) {
        this.cloudinaryUploader = cloudinaryUploader;
    }

    @PostMapping("/upload")
    public String uploadImage(@RequestBody String base64Data) throws IOException {
        return cloudinaryUploader.uploadBase64Image(base64Data);
    }
}