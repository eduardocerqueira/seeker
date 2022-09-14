//date: 2022-09-14T17:20:10Z
//url: https://api.github.com/gists/15590f4711c7f2d86786ebf07e9b9118
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Image;
import com.aspose.imaging.imageoptions.TgaOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

try (Image image = Image.load(dataDir + "template.jpg"))
{
	image.save(dataDir + "result.tga", new TgaOptions());
}

new File(dataDir + "result.tga").delete();
