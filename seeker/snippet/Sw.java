//date: 2023-08-18T17:08:05Z
//url: https://api.github.com/gists/222865023486a7f313d69524e7efcb71
//owner: https://api.github.com/users/AkashSutar02

package sw;

import java.awt.BorderLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

class C1 implements ActionListener {
    private static JTextField inputBox;
    public JComponent frmCalculator;


    C1(){}
    public static void main(String[] args) {
        createWindow();
    }

    private static void createWindow() {
        JFrame frame = new JFrame("Calculator");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        createUI(frame);
        frame.setSize(200, 200);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private static void createUI(JFrame frame) {
        JPanel panel = new JPanel();
        C1 c1 = new C1();
        GridBagLayout layout = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        panel.setLayout(layout);

        inputBox = new JTextField(10);
        inputBox.setEditable(false);

        JButton button0 = new JButton("0");JButton button1 = new JButton("1");
        JButton button2 = new JButton("2");JButton button3 = new JButton("3");
        JButton button4 = new JButton("4");JButton button5 = new JButton("5");
        JButton button6 = new JButton("6");JButton button7 = new JButton("7");
        JButton button8 = new JButton("8");JButton button9 = new JButton("9");

        JButton buttonPlus = new JButton("+");JButton buttonMinus = new JButton("-");
        JButton buttonDivide = new JButton("/");JButton buttonMultiply = new JButton("x");
        JButton buttonClear = new JButton("C");JButton buttonDot = new JButton(".");
        JButton buttonEquals = new JButton("=");

        button1.addActionListener(c1);button2.addActionListener(c1);
        button3.addActionListener(c1);button4.addActionListener(c1);
        button5.addActionListener(c1);button6.addActionListener(c1);
        button7.addActionListener(c1);button8.addActionListener(c1);
        button9.addActionListener(c1);button0.addActionListener(c1);

        buttonPlus.addActionListener(c1);buttonMinus.addActionListener(c1);
        buttonDivide.addActionListener(c1);buttonMultiply.addActionListener(c1);
        buttonClear.addActionListener(c1);buttonDot.addActionListener(c1);
        buttonEquals.addActionListener(c1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.gridx = 0; gbc.gridy = 0; panel.add(button1, gbc);
        gbc.gridx = 1; gbc.gridy = 0; panel.add(button2, gbc);
        gbc.gridx = 2; gbc.gridy = 0; panel.add(button3, gbc);
        gbc.gridx = 3; gbc.gridy = 0; panel.add(buttonPlus, gbc);
        gbc.gridx = 0; gbc.gridy = 1; panel.add(button4, gbc);
        gbc.gridx = 1; gbc.gridy = 1; panel.add(button5, gbc);
        gbc.gridx = 2; gbc.gridy = 1; panel.add(button6, gbc);
        gbc.gridx = 3; gbc.gridy = 1; panel.add(buttonMinus, gbc);
        gbc.gridx = 0; gbc.gridy = 2; panel.add(button7, gbc);
        gbc.gridx = 1; gbc.gridy = 2; panel.add(button8, gbc);
        gbc.gridx = 2; gbc.gridy = 2; panel.add(button9, gbc);
        gbc.gridx = 3; gbc.gridy = 2; panel.add(buttonDivide, gbc);
        gbc.gridx = 0; gbc.gridy = 3; panel.add(buttonDot, gbc);
        gbc.gridx = 1; gbc.gridy = 3; panel.add(button0, gbc);
        gbc.gridx = 2; gbc.gridy = 3; panel.add(buttonClear, gbc);
        gbc.gridx = 3; gbc.gridy = 3; panel.add(buttonMultiply, gbc);
        gbc.gridwidth = 3;

        gbc.gridx = 0; gbc.gridy = 4; panel.add(inputBox, gbc);
        gbc.gridx = 3; gbc.gridy = 4; panel.add(buttonEquals, gbc);
        frame.getContentPane().add(panel, BorderLayout.CENTER);
    }

    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        if (command.charAt(0) == 'C') {
            inputBox.setText("");
        }else if (command.charAt(0) == '=') {
            inputBox.setText(evaluate(inputBox.getText()));
        }else {
            inputBox.setText(inputBox.getText() + command);
        }
    }

    public static String evaluate(String expression) {
        char[] arr = expression.toCharArray();
        String operand1 = "";String operand2 = "";String operator = "";
        double result = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= '0' && arr[i] <= '9' || arr[i] == '.') {
                if(operator.isEmpty()){
                    operand1 += arr[i];
                }else{
                    operand2 += arr[i];
                }
            }

            if(arr[i] == '+' || arr[i] == '-' || arr[i] == '/' || arr[i] == '*') {
                operator += arr[i];
            }
        }

        if (operator.equals("+"))
            result = (Double.parseDouble(operand1) + Double.parseDouble(operand2));
        else if (operator.equals("-"))
            result = (Double.parseDouble(operand1) - Double.parseDouble(operand2));
        else if (operator.equals("/"))
            result = (Double.parseDouble(operand1) / Double.parseDouble(operand2));
        else
            result = (Double.parseDouble(operand1) * Double.parseDouble(operand2));
        return operand1 + operator + operand2 + "=" +result;
    }
}
