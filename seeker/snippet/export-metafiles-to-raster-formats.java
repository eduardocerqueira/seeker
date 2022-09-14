//date: 2022-09-14T17:20:06Z
//url: https://api.github.com/gists/f8690c34968d631d7673a25f4a1e5daf
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.Image;
import com.aspose.imaging.coreexceptions.ImageLoadException;
import com.aspose.imaging.fileformats.emf.EmfImage;
import com.aspose.imaging.imageoptions.BmpOptions;
import com.aspose.imaging.imageoptions.EmfRasterizationOptions;
import com.aspose.imaging.imageoptions.GifOptions;
import com.aspose.imaging.imageoptions.JpegOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

// Create EmfRasterizationOption class instance and set properties
EmfRasterizationOptions emfRasterizationOptions = new EmfRasterizationOptions();
emfRasterizationOptions.setBackgroundColor(Color.getPapayaWhip());
emfRasterizationOptions.setPageWidth(300);
emfRasterizationOptions.setPageHeight(300);

// Load an existing EMF file as image and convert it to EmfImage class object
try (EmfImage image = (EmfImage) Image.load(dataDir + "template.emf"))
{
	if (!image.getHeader().getEmfHeader().getValid())
	{
		throw new ImageLoadException(String.format("The file %s is not valid", dataDir + "Picture1.emf"));
	}

	// Convert EMF to BMP, GIF, JPEG, J2K, PNG, PSD, TIFF and WebP
	image.save(dataDir + "result.bmp", new BmpOptions() {{
			setVectorRasterizationOptions(emfRasterizationOptions);
		}});
	image.save(dataDir + "result.gif", new GifOptions() {{
			setVectorRasterizationOptions(emfRasterizationOptions);
		}});
	image.save(dataDir + "result.jpg", new JpegOptions() {{
			setVectorRasterizationOptions(emfRasterizationOptions);
		}});
}

new File(dataDir + "result.bmp").delete();
new File(dataDir + "result.gif").delete();
new File(dataDir + "result.jpg").delete();
