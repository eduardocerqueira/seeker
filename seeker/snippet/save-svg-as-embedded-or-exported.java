//date: 2022-09-14T17:20:06Z
//url: https://api.github.com/gists/f8690c34968d631d7673a25f4a1e5daf
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.Image;
import com.aspose.imaging.coreexceptions.FrameworkException;
import com.aspose.imaging.fileformats.svg.SvgResourceKeeperCallback;
import com.aspose.imaging.imageoptions.EmfRasterizationOptions;
import com.aspose.imaging.imageoptions.SvgOptions;
import com.aspose.ms.System.InvalidOperationException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

saveWithEmbeddedImages();
//saveWithExportImages();
new File(dataDir + "result.png").delete();


static void saveWithEmbeddedImages()
{
	String[] files = {dataDir + "template.svg"};
	for (String file : files)
	{
		save(true, file, null);
	}
}

static void saveWithExportImages()
{
	String[] files = {dataDir + "template.svg"};
	String[][] expectedImages =
			{
					{
							"result.png"
					},
			};

	for (int i = 0; i < files.length; i++)
	{
		save(false, files[i], expectedImages[i]);
	}
}

static void save(boolean useEmbedded, String inputFile, String[] expectedImages)
{
	String outputFile = dataDir + "result.svg";
	String imageFolder;
	try (Image image = Image.load(inputFile))
	{
		EmfRasterizationOptions emfRasterizationOptions = new EmfRasterizationOptions();
		emfRasterizationOptions.setBackgroundColor(Color.getWhite());
		emfRasterizationOptions.setPageWidth(image.getWidth());
		emfRasterizationOptions.setPageHeight(image.getHeight());
		String testingFileName = new File(inputFile).getName();
		testingFileName = testingFileName.substring(0, testingFileName.lastIndexOf("."));
		imageFolder = dataDir + testingFileName;

		SvgCallbackImageTest callback = new SvgCallbackImageTest(useEmbedded, imageFolder)
		{{
			setLink("Images/" + testingFileName);
		}};

		image.save(outputFile,
				new SvgOptions()
				{{
					setVectorRasterizationOptions(emfRasterizationOptions);
					setCallback(callback);
				}});
	}

	if (!useEmbedded)
	{
		String[] files = new File(dataDir + "Images").list();
		if (files == null || files.length != expectedImages.length)
		{
			throw new AssertionError(String.format("Expected count font files = %d, Current count image files = %d",
					expectedImages.length,
					files == null ? 0 : files.length));
		}

		for (int i = 0; i < files.length; i++)
		{
			String file = files[i];

			if (!file.equalsIgnoreCase(expectedImages[i]))
			{
				throw new AssertionError(String.format("Expected file name: '%s', current: '%s'", expectedImages[i], file));
			}
		}
	}
}

static class SvgCallbackImageTest extends SvgResourceKeeperCallback
{
	/**
	 * <p>
	 * The out folder
	 * </p>
	 */
	private final String outFolder;

	/**
	 * <p>
	 * The use embedded font
	 * </p>
	 */
	private final boolean useEmbeddedImage;

	/**
	 * <p>Initializes a new instance of the class.</p>
	 *
	 * @param useEmbeddedImage if set to <c>true</c> [use embedded image].
	 * @param outFolder        The out folder.
	 */
	public SvgCallbackImageTest(boolean useEmbeddedImage, String outFolder)
	{
		this.useEmbeddedImage = useEmbeddedImage;
		this.outFolder = outFolder;
	}

	private String link;

	public void setLink(String link)
	{
		this.link = link;
	}

	/**
	 * <p>
	 * Called when image resource ready.
	 * </p>
	 *
	 * @param imageData         The resource data.
	 * @param imageType         Type of the image.
	 * @param suggestedFileName Name of the suggested file.
	 * @param useEmbeddedImage  if set to <c>true</c> the embedded image must be used.
	 * @return Returns path to saved resource. Path should be relative to target SVG document.
	 */
	@Override
	public String onImageResourceReady(byte[] imageData, int imageType, String suggestedFileName, boolean[] useEmbeddedImage)
	{
		useEmbeddedImage[0] = this.useEmbeddedImage;

		if (useEmbeddedImage[0])
		{
			return suggestedFileName;
		}

		String fontFolder = this.outFolder;
		final File dir = new File(fontFolder);
		if (!dir.exists() || !dir.isDirectory())
		{
			if (!dir.mkdirs())
			{
				throw new InvalidOperationException("Error creating a directory: " + dir.getAbsolutePath());
			}
		}

		String fileName = fontFolder + "\\" + new File(suggestedFileName).getName();

		try (FileOutputStream fs = new FileOutputStream(fileName))
		{
			fs.write(imageData, 0, imageData.length);
		}
		catch (IOException e)
		{
			throw new FrameworkException(e.getMessage(), e);
		}

		return "./" + this.link + "/" + suggestedFileName;
	}
}
