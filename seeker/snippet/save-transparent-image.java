//date: 2022-02-10T16:50:00Z
//url: https://api.github.com/gists/0abc4583f4466b655fd5c35925c41dbe
//owner: https://api.github.com/users/ESidenko

import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.png.PngColorType;
import com.aspose.imaging.imageoptions.PngOptions;

String templatesFolder = "c:\\Users\\USER\\Downloads\\";

String inputFile = templatesFolder + "template.png";
String outputFile = templatesFolder + "output.png";

try (Image image = Image.load(inputFile))
{
	image.save(outputFile, new PngOptions()
	{{
		setColorType(PngColorType.TruecolorWithAlpha);
	}});
}
