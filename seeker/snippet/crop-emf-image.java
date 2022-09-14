//date: 2022-09-14T17:20:06Z
//url: https://api.github.com/gists/f8690c34968d631d7673a25f4a1e5daf
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.emf.EmfImage;
import com.aspose.imaging.imageoptions.EmfRasterizationOptions;
import com.aspose.imaging.imageoptions.PdfOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

// Create an instance of Rasterization options
EmfRasterizationOptions emfRasterizationOptions = new EmfRasterizationOptions();
emfRasterizationOptions.setBackgroundColor(Color.getWhiteSmoke());

// Create an instance of PNG options
PdfOptions pdfOptions = new PdfOptions();
pdfOptions.setVectorRasterizationOptions(emfRasterizationOptions);

// Load an existing image into an instance of EMF class
try (EmfImage image = (EmfImage) Image.load(dataDir + "template.emf"))
{
	// Based on the shift values, apply the cropping on image and Crop method will shift the image bounds toward the center of image
	image.crop(0, 0, 30, 30);

	// Set height and width and  Save the results to disk
	emfRasterizationOptions.setPageWidth(image.getWidth());
	emfRasterizationOptions.setPageHeight(image.getHeight());
	image.save(dataDir + "result.pdf", pdfOptions);
}
new File(dataDir + "result.pdf").delete();
