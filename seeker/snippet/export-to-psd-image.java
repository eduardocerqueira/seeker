//date: 2022-09-14T17:20:08Z
//url: https://api.github.com/gists/fe8c8c8a9ba2c9680f9479b43d8c0498
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.psd.ColorModes;
import com.aspose.imaging.fileformats.psd.CompressionMethod;
import com.aspose.imaging.imageoptions.PsdOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

// Load an existing image
try (Image image = Image.load(dataDir + "template.bmp"))
{
	// Create an instance of PsdOptions, Set its various properties Save image to disk in PSD format
	PsdOptions psdOptions = new PsdOptions();
	psdOptions.setColorMode(ColorModes.Rgb);
	psdOptions.setCompressionMethod(CompressionMethod.Raw);
	psdOptions.setVersion(4);
	image.save(dataDir + "ExportImageToPSD_output.psd", psdOptions);
}

new File(dataDir + "ExportImageToPSD_output.psd").delete();
