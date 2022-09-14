//date: 2022-09-14T17:20:06Z
//url: https://api.github.com/gists/f8690c34968d631d7673a25f4a1e5daf
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.FontSettings;
import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.emf.EmfImage;
import com.aspose.imaging.imageoptions.EmfRasterizationOptions;
import com.aspose.imaging.imageoptions.PngOptions;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

// Create EmfRasterizationOption class instance and set properties
EmfRasterizationOptions emfRasterizationOptions = new EmfRasterizationOptions();
emfRasterizationOptions.setBackgroundColor(Color.getWhiteSmoke());

// Create an instance of PNG options
PngOptions pngOptions = new PngOptions();
pngOptions.setVectorRasterizationOptions(emfRasterizationOptions);

// Load an existing EMF image
try (EmfImage image = (EmfImage) Image.load(dataDir + "template.emf"))
{
	image.cacheData();

	// Set height and width, Reset font settings
	emfRasterizationOptions.setPageWidth(300);
	emfRasterizationOptions.setPageHeight(350);
	FontSettings.reset();
	image.save(dataDir + "result.png", pngOptions);

	// Initialize font list
	List<String> fonts = new ArrayList<>();
	Collections.addAll(fonts, FontSettings.getDefaultFontsFolders());

	// Add new font path to font list and Assign list of font folders to font settings and Save the EMF file to PNG image with new font
	fonts.add(dataDir + "fonts\\");
	FontSettings.setFontsFolders(fonts.toArray(new String[0]), true);
	image.save(dataDir + "result2.png", pngOptions);
}

new File(dataDir + "result.png").delete();
new File(dataDir + "result2.png").delete();
