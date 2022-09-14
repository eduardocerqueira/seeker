//date: 2022-09-14T17:20:08Z
//url: https://api.github.com/gists/fe8c8c8a9ba2c9680f9479b43d8c0498
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.eps.EpsBinaryImage;
import com.aspose.imaging.fileformats.eps.EpsImage;
import com.aspose.imaging.fileformats.eps.EpsInterchangeImage;
import com.aspose.imaging.fileformats.eps.consts.EpsType;
import com.aspose.imaging.imageoptions.PngOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

try (EpsImage epsImage = (EpsImage) Image.load(dataDir + "template.eps"))
{
	// check if EPS image has any raster preview to proceed (for now only raster preview is supported)
	if (epsImage.hasRasterPreview())
	{
		if (epsImage.getPhotoshopThumbnail() != null)
		{
			// process Photoshop thumbnail if it's present
		}

		if (epsImage.getEpsType() == EpsType.Interchange)
		{
			// Get EPS Interchange subformat instance
			EpsInterchangeImage epsInterchangeImage = (EpsInterchangeImage) epsImage;

			if (epsInterchangeImage.getRasterPreview() != null)
			{
				// process black-and-white Interchange raster preview if it's present
			}
		}
		else
		{
			// Get EPS Binary subformat instance
			EpsBinaryImage epsBinaryImage = (EpsBinaryImage) epsImage;

			if (epsBinaryImage.getTiffPreview() != null)
			{
				// process TIFF preview if it's present
			}

			if (epsBinaryImage.getWmfPreview() != null)
			{
				// process WMF preview if it's present
			}
		}

		// export EPS image to PNG (by default, best available quality preview is used for export)
		epsImage.save(dataDir + "anyEpsFile.png", new PngOptions());
	}
}

new File(dataDir + "anyEpsFile.png  ").delete();
