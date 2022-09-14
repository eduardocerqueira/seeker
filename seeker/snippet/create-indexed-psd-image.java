//date: 2022-09-14T17:20:08Z
//url: https://api.github.com/gists/fe8c8c8a9ba2c9680f9479b43d8c0498
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.*;
import com.aspose.imaging.fileformats.psd.ColorModes;
import com.aspose.imaging.fileformats.psd.CompressionMethod;
import com.aspose.imaging.imageoptions.PsdOptions;
import com.aspose.imaging.sources.FileCreateSource;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

try (PsdOptions createOptions = new PsdOptions())
{
	createOptions.setSource(new FileCreateSource(dataDir + "Newsample_out.psd", false));
	createOptions.setColorMode(ColorModes.Indexed);
	createOptions.setVersion(5);

	// Create a new color palette having RGB colors, Set Palette property & compression method
	Color[] palette = {Color.getRed(), Color.getGreen(), Color.getBlue()};
	createOptions.setPalette(new ColorPalette(palette));
	createOptions.setCompressionMethod(CompressionMethod.RLE);

	// Create a new PSD with PsdOptions created previously
	try (Image psd = Image.create(createOptions, 500, 500))
	{
		// Draw some graphics over the newly created PSD
		Graphics graphics = new Graphics(psd);
		graphics.clear(Color.getWhite());
		graphics.drawEllipse(new Pen(Color.getRed(), 6), new Rectangle(0, 0, 400, 400));
		psd.save();
	}
}

new File(dataDir + "Newsample_out.psd").delete();
