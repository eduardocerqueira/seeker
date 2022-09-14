//date: 2022-09-14T17:20:06Z
//url: https://api.github.com/gists/f8690c34968d631d7673a25f4a1e5daf
//owner: https://api.github.com/users/aspose-com-gists

import com.aspose.imaging.Color;
import com.aspose.imaging.Image;
import com.aspose.imaging.coreexceptions.ImageLoadException;
import com.aspose.imaging.fileformats.emf.EmfImage;
import com.aspose.imaging.imageoptions.EmfRasterizationOptions;
import com.aspose.imaging.imageoptions.PdfOptions;
import java.io.File;



String dataDir = "c:\\Users\\USER\\Downloads\\templates\\";

String[] filePaths = {
		"template.emf"
};

for (String filePath : filePaths)
{
	String outPath = dataDir + "result.pdf";
	try (EmfImage image = (EmfImage) Image.load(dataDir + filePath))
	{
		if (!image.getHeader().getEmfHeader().getValid())
		{
			throw new ImageLoadException(String.format("The file %s is not valid", outPath));
		}

		EmfRasterizationOptions emfRasterization = new EmfRasterizationOptions();
		emfRasterization.setPageWidth(image.getWidth());
		emfRasterization.setPageHeight(image.getHeight());
		emfRasterization.setBackgroundColor(Color.getWhiteSmoke());
		PdfOptions pdfOptions = new PdfOptions();
		pdfOptions.setVectorRasterizationOptions(emfRasterization);
		image.save(outPath, pdfOptions);
	}
}

new File(dataDir + "result.pdf").delete();
