//date: 2023-11-21T17:02:32Z
//url: https://api.github.com/gists/0ed9c2710c12fbbc1fe2bb4dfffe6629
//owner: https://api.github.com/users/halitkalayci

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;

import java.io.IOException;
import java.util.Map;

public class CloudinaryUploader {
    
    private Cloudinary cloudinary;

    public CloudinaryUploader(String cloudName, String apiKey, String apiSecret) {
        cloudinary = new Cloudinary(ObjectUtils.asMap(
                "cloud_name", cloudName,
                "api_key", apiKey,
                "api_secret", apiSecret));
    }

    public String uploadBase64Image(String base64Data) throws IOException {
        Map<?, ?> uploadResult = cloudinary.uploader().upload(base64Data, ObjectUtils.emptyMap());
        return (String) uploadResult.get("url");
    }
}