//date: 2024-11-22T16:57:21Z
//url: https://api.github.com/gists/a4016401658f15de7b3dabe2ad8d7c92
//owner: https://api.github.com/users/dxenes1

import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;
import ij.gui.GenericDialog;
import ij.plugin.frame.RoiManager;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.FileSaver;
import java.io.File;
import java.io.IOException;

// AWS SDK imports
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.AmazonS3;

public class ExportAndUploadPlugin implements PlugIn {

    @Override
    public void run(String arg) {
        // Get the current image
        ImagePlus imp = IJ.getImage();
        if (imp == null) {
            IJ.error("No image is open.");
            return;
        }

        // Get ROIs from the ROI Manager and add them to the image overlay
        RoiManager roiManager = RoiManager.getInstance();
        if (roiManager != null && roiManager.getCount() > 0) {
            Overlay overlay = new Overlay();
            for (Roi roi : roiManager.getRoisAsArray()) {
                overlay.add(roi);
            }
            imp.setOverlay(overlay);
        } else {
            IJ.showMessage("No ROIs found in the ROI Manager.");
        }

        // Prompt user for AWS credentials and bucket information
        GenericDialog gd = new GenericDialog("Upload to S3");
        gd.addStringField("AWS Access Key ID:", "");
        gd.addStringField("AWS Secret Access Key: "**********"
        gd.addStringField("AWS Region:", "us-east-1");
        gd.addStringField("S3 Bucket Name:", "");
        gd.addStringField("S3 Object Key (File Name):", imp.getTitle() + ".tiff");
        gd.showDialog();
        if (gd.wasCanceled()) {
            return;
        }

        String accessKeyId = gd.getNextString().trim();
        String secretAccessKey = "**********"
        String region = gd.getNextString().trim();
        String bucketName = gd.getNextString().trim();
        String objectKey = gd.getNextString().trim();

        if (accessKeyId.isEmpty() || secretAccessKey.isEmpty() || bucketName.isEmpty() || objectKey.isEmpty()) {
            IJ.error("All fields are required.");
            return;
        }

        // Save the image with ROIs as a TIFF file to a temporary location
        File tempFile = null;
        try {
            tempFile = File.createTempFile("imagej_export_", ".tiff");
            FileSaver fileSaver = new FileSaver(imp);
            fileSaver.saveAsTiff(tempFile.getAbsolutePath());
        } catch (IOException e) {
            IJ.error("Error saving the image: " + e.getMessage());
            return;
        }

        // Upload the file to the specified S3 bucket
        try {
            BasicAWSCredentials awsCreds = "**********"
            AmazonS3 s3Client = AmazonS3ClientBuilder.standard()
                    .withRegion(region)
                    .withCredentials(new AWSStaticCredentialsProvider(awsCreds))
                    .build();

            s3Client.putObject(bucketName, objectKey, tempFile);
            IJ.showMessage("Upload Successful", "Image uploaded to S3 bucket: " + bucketName);
        } catch (Exception e) {
            IJ.error("Error uploading to S3: " + e.getMessage());
        } finally {
            // Clean up the temporary file
            if (tempFile != null && tempFile.exists()) {
                tempFile.delete();
            }
        }
    }
}
empFile.delete();
            }
        }
    }
}
