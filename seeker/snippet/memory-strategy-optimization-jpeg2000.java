//date: 2022-09-14T17:20:05Z
//url: https://api.github.com/gists/47931bb418e53a9eba11bc1cccad5863
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Image;
import com.aspose.imaging.ImageOptionsBase;
import com.aspose.imaging.LoadOptions;
import com.aspose.imaging.fileformats.jpeg2000.Jpeg2000Codec;
import com.aspose.imaging.imageoptions.Jpeg2000Options;
import com.aspose.imaging.sources.FileCreateSource;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

// Setting a memory limit of 10 megabytes for target loaded image
// JP2 codec
try (Image image = Image.load(dataDir + "couple.jp2", new LoadOptions() {{ setBufferSizeHint(100); }}))
{
	image.save(dataDir + "result.jp2");
}

new File(dataDir + "result.jp2").delete();

// Setting a memory limit of 10 megabytes for target created image
// JP2 codec
try (ImageOptionsBase createOptions = new Jpeg2000Options() {{ setCodec(Jpeg2000Codec.Jp2); }})
{
	createOptions.setBufferSizeHint(10);
	createOptions.setSource(new FileCreateSource(dataDir + "result.jp2", false));
	try (Image image = Image.create(createOptions, 100, 100))
	{
		image.save(); // save to same location
	}
}

new File(dataDir + "result.jp2").delete();

// J2K codec
try (ImageOptionsBase createOptions = new Jpeg2000Options() {{ setCodec(Jpeg2000Codec.J2K); }})
{
	createOptions.setBufferSizeHint(10);
	createOptions.setSource(new FileCreateSource(dataDir + "result.j2k", false));
	try (Image image = Image.create(createOptions, 100, 100))
	{
		image.save(); // save to same location
	}
}

new File(dataDir + "result.j2k").delete();
