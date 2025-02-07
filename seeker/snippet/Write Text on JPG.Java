//date: 2025-02-07T16:56:52Z
//url: https://api.github.com/gists/d707f5b6a144e239bbe44f01386ff400
//owner: https://api.github.com/users/aspose-com-kb

// Load the image
com.aspose.drawing.Bitmap bitmap = new com.aspose.drawing.Bitmap("input.jpg"); 
com.aspose.drawing.Graphics graphics = com.aspose.drawing.Graphics.fromImage(bitmap); 

// Define text color
com.aspose.drawing.SolidBrush brush = new com.aspose.drawing.SolidBrush(com.aspose.drawing.Color.getDarkBlue()); 

// Define text font
com.aspose.drawing.Font arial = new com.aspose.drawing.Font("Arial", 34, com.aspose.drawing.FontStyle.Regular); 

// Text to display
String text = "Sample Text"; 

// Define rectangle
com.aspose.drawing.RectangleF rect = new com.aspose.drawing.RectangleF(100, 600, 850, 100); 

// Draw text on image 
graphics.drawString(text, arial, brush, rect); 
 
// Save the output file 
bitmap.save("output.jpg"); 
