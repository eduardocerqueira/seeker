//date: 2022-09-14T17:20:10Z
//url: https://api.github.com/gists/15590f4711c7f2d86786ebf07e9b9118
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Image;
import com.aspose.imaging.RasterImage;
import com.aspose.imaging.fileformats.tga.TgaImage;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

try (RasterImage image = (RasterImage) Image.load(dataDir + "template.png"))
{
	try (TgaImage tgaImage = new TgaImage(image))
	{
		tgaImage.save(dataDir + "result.tga");
	}
}

new File(dataDir + "result.tga").delete();
