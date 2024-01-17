//date: 2024-01-17T16:58:47Z
//url: https://api.github.com/gists/11374e4c96f906c5291e39e3be73438e
//owner: https://api.github.com/users/mirko0

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.net.URL;
import javax.imageio.*;

public class ChangeMyMindGenerator {

    // Define a method to generate the meme image
    public BufferedImage generateMeme(String text) {
        try {
            text = splitText(text, 30);
            // Load the meme template image from a file
            BufferedImage template = ImageIO.read(new URL("https://i.imgur.com/j0NdJYG.jpg"));
            //BufferedImage template = ImageIO.read(new File("changemymind.jpg"));
            // Get the width and height of the template image
            int width = template.getWidth();
            int height = template.getHeight();

            // Create a new image with the same size as the template image
            BufferedImage meme = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            // Get the graphics object of the new image
            Graphics2D g = meme.createGraphics();
            // Draw the template image on the new image
            g.drawImage(template, 0, 0, null);

            // Font Properties
            Font font = new Font("Arial", Font.BOLD, 35);
            g.setFont(font);
            g.setColor(Color.BLACK);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            // Get the font metrics for the text
            FontMetrics fm = g.getFontMetrics();

            int x = 270;
            int y = 600; // Initial y-coordinate

            String[] lines = text.split("\n");
            // Draw each line of text on the new image
            for (int i = 0; i < lines.length; i++) {
                // Draw only 5 lines max
                if (i == 5) break;
                // Draw the text on the new image
                g.drawString(lines[i], x, y);
                // Move to the next line
                y += fm.getHeight();
            }


            // Dispose the graphics object
            g.dispose();
            // Save the new image to a file
            ImageIO.write(meme, "jpg", new File("meme.jpg"));

            return meme;
        } catch (IOException e) {
            // Print an error message
            System.out.println("An error occurred while generating chnagemymind meme: " + e.getMessage());
            return null;
        }
    }
    
    
    public String splitText(String text, int maxLength) {
        StringBuilder result = new StringBuilder();
        int lineLength = 0;

        // Split string in to words
        String[] words = text.split("\\s+");

        // Append the words until their combine length reaches max length
        for (String word : words) {
            if (lineLength + word.length() > maxLength) {
                result.append("\n");
                lineLength = 0;
            }
            if (lineLength > 0) {
                result.append(" ");
                lineLength++;
            }
            result.append(word);
            lineLength += word.length();
        }

        return result.toString();
    }


}
