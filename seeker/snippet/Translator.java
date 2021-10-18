//date: 2021-10-18T17:06:25Z
//url: https://api.github.com/gists/ad0ad6e6e5543415debb135712346621
//owner: https://api.github.com/users/anelaco

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Scanner;

public class Translator extends JFrame implements ActionListener {
    public static final int HEIGHT = 100;
    public static final int WIDTH = 1200;
    public static final int SIZE = 30;
    public JTextField input;
    public JTextField output;
    public String textI = "Enter text here";

    public Translator(){
        super("Pig Latin Translator");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new FlowLayout(FlowLayout.CENTER));

        JPanel inputPanel = new JPanel();
        inputPanel.setLayout(new BorderLayout());
        inputPanel.setBackground(Color.MAGENTA);

        input = new JTextField(SIZE);
        input.setText(textI);
        input.setBounds(10,20,80,25);
        inputPanel.add(input, BorderLayout.EAST);

        JLabel inputLabel = new JLabel("Input text here: ");
        inputLabel.setBounds(10,20,80,25 );
        inputPanel.add(inputLabel, BorderLayout.CENTER);
        add(inputPanel);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new BorderLayout());
        buttonPanel.setBackground(Color.MAGENTA);

        JButton translate = new JButton("Translate");
        translate.setBounds(20,30,20,30);
        translate.setBackground(Color.LIGHT_GRAY);
        translate.addActionListener(this);
        buttonPanel.add(translate, BorderLayout.CENTER);
        add(buttonPanel);

        JPanel outputPanel = new JPanel();
        outputPanel.setLayout(new BorderLayout());
        outputPanel.setBackground(Color.MAGENTA);

        output = new JTextField(SIZE);
        output.setText("");
        outputPanel.add(output, BorderLayout.EAST);

        JLabel outputLabel = new JLabel("The translation: ");
        outputLabel.setBounds(10,20,80,25 );
        outputPanel.add(outputLabel, BorderLayout.CENTER);
        add(outputPanel);
    }

    public void actionPerformed(ActionEvent e) {
        String actionCommand = e.getActionCommand( );
        char[] arV = {'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U'};
        char[] arP = {'.', ',',';','?','!','"'};
        int j = 0;
        StringBuilder totalString = new StringBuilder();
        if(actionCommand.equals("Translate")) {
            Scanner scan = new Scanner(input.getText());

            while (scan.hasNext()){
                String newString ="";
                boolean con = false;
                String thisWord = scan.next();

                for (char value : arP) {
                    if (thisWord.charAt(j) == value) {
                        thisWord = thisWord.substring(1);
                    }
                    if (thisWord.charAt(thisWord.length() - 1) == value) {
                        thisWord = thisWord.substring(0, thisWord.length() - 1);
                    }
                }

                for (char c : arV) {
                    if (thisWord.charAt(j) == c) {
                        con = true;
                    }
                }

                if (con) {
                    newString = thisWord + "way  ";
                    totalString.append(newString);
                }else{
                    newString = thisWord.substring(1) + thisWord.charAt(j) + "ay ";
                    totalString.append(newString);
                }
            }
            output.setText(String.valueOf(totalString));
        }
    }

    public static void main(String[] args)
    {
        Translator myTranslator = new Translator();
        myTranslator.setVisible(true);
    }
}