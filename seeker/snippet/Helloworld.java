//date: 2022-03-10T16:59:04Z
//url: https://api.github.com/gists/850d6a26a585fc4fb2763579d6d36c46
//owner: https://api.github.com/users/visiplusUser1

import java.awt.Color;
import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

public class HelloWorld implements Constants {

        public static void main(String[] args) {

                JFrame frame = new JFrame(HELLO_WORLD_JAVA_SWING);

                // set frame site
                frame.setMinimumSize(new Dimension(200, 200));
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

                // center the JLabel
                JLabel lblText = new JLabel(HELLO_WORLD, SwingConstants.CENTER);
                Color yellow = new Color(240, 240, 20);
                lblText.setOpaque(true);
                lblText.setBackground(yellow);

                // add JLabel to JFrame
                frame.getContentPane().add(lblText);

                // display it
                frame.pack();
                frame.setVisible(true);

        }
}