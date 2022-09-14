//date: 2022-09-14T17:20:10Z
//url: https://api.github.com/gists/15590f4711c7f2d86786ebf07e9b9118
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.Image;
import com.aspose.imaging.fileformats.tga.TgaImage;
import java.io.File;
import java.util.Calendar;
import java.util.Date;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

try (TgaImage image = (TgaImage) Image.load(dataDir + "template.tga"))
{
	image.setDateTimeStamp(new Date());
	image.setAuthorName("John Smith");
	image.setAuthorComments("Comment");
	image.setImageId("ImageId");
	image.setJobNameOrId("Important Job");
	image.setJobTime(new Date(0, Calendar.JANUARY, 10));
	image.setTransparentColor(Color.fromArgb(123));
	image.setSoftwareId("SoftwareId");
	image.setSoftwareVersion("abc1");
	image.setSoftwareVersionLetter('a');
	image.setSoftwareVersionNumber(2);
	image.setXOrigin(1000);
	image.setYOrigin(1000);

	image.save(dataDir + "result.tga");
}

new File(dataDir + "result.tga").delete();
