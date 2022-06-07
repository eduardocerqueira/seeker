//date: 2022-06-07T17:00:41Z
//url: https://api.github.com/gists/162986c284168e832916834238e20055
//owner: https://api.github.com/users/ai-null

import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;
import java.awt.EventQueue;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Timer extends JFrame {
    static JLabel jLabelInput;
    static JLabel jLabelHour;
    static JLabel jLabelMinute;
    static JLabel jLabelSecond;
    static JButton jButtonStart;
    static JButton jButtonReset;
    static JTextField jTFTimerInput;

    static Thread t;

    static int mil = 0;
    static int second = 0;
    static int minute = 0;

    static Boolean isStart = false;

    public Timer() {
        initComponents();
    }

    public void initComponents() {
        jLabelInput = new JLabel("MASUKKAN WAKTU DALAM MENIT");
        jLabelHour = new JLabel("00 :");
        jLabelMinute = new JLabel("00 :");
        jLabelSecond = new JLabel("00");
        jButtonStart = new JButton();
        jButtonReset = new JButton();
        jTFTimerInput = new JTextField(3);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        jButtonStart.setText("Start");
        jButtonStart.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                startTimer();
            }
        });

        jButtonReset.setText("reset");
        jButtonReset.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                stopTimer();
            }
        });

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);

        layout.setHorizontalGroup(
            layout.createParallelGroup(Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(Alignment.LEADING)
                .addGroup(layout.createSequentialGroup()
                    .addGap(17, 17, 17)
                    .addComponent(jLabelInput)
                    .addGap(17, 17, 17)
                    .addComponent(jTFTimerInput))
                .addGroup(layout.createSequentialGroup()
                    .addGap(36, 36, 36)
                    .addComponent(jLabelHour)
                    .addGap(18, 18, 18)
                    .addComponent(jLabelMinute)
                    .addGap(18, 18, 18)
                    .addComponent(jLabelSecond))
                .addGroup(layout.createSequentialGroup()
                    .addGap(17, 17, 17)
                    .addComponent(jButtonStart)
                    .addPreferredGap(ComponentPlacement.UNRELATED)
                    .addComponent(jButtonReset)))
            .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jLabelInput)
                    .addComponent(jTFTimerInput))
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jLabelHour)
                    .addComponent(jLabelMinute)
                    .addComponent(jLabelSecond))
                .addGap(30, 30, 30)
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jButtonStart)
                    .addComponent(jButtonReset))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        pack();
    }

    private void startTimer() {
        setTimer();

        isStart = true;
        t = new Thread() {
            public void run () {
                while (isStart) {
                    try {
                        sleep(1);
                        
                        if (0 >= mil) {
                            mil = 1000;
                            second--;
                        }

                        if (0 > second) {
                            second = 60;
                            minute--;
                        }

                        if (0 > minute) {
                            break;
                        }

                        mil--;
                        jLabelSecond.setText(mil+"  :");
                        jLabelMinute.setText(second+" :");
                        jLabelHour.setText(minute+" :");
                    } catch (Exception e) {
                        handleException(e);
                    }
                }
            }
        };
        t.start();
    }

    private void stopTimer() {
        t.stop();

        isStart = false;
        minute = 0;
        second = 0;
        mil = 0;

        jLabelHour.setText("00 :");
        jLabelMinute.setText("00 :");
        jLabelSecond.setText("00");
    }

    private void setTimer() {
        String timeInString = jTFTimerInput.getText();

        if (timeInString.isBlank()) {
            handleException(new Exception("Tolong inputkan waktu"));
            return;
        }

        try {
            int timeInHour = Integer.parseInt(timeInString);
            minute = timeInHour;
            second = 0;
            mil = 0;
        } catch (Exception e) {
            handleException(e);
        }
    }

    private void handleException(Exception e) {
        JOptionPane.showMessageDialog(null, e.getMessage(), "Snap!", JOptionPane.ERROR_MESSAGE);
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(new Runnable() {
           @Override
           public void run() {
               new Timer().setVisible(true);
           } 
        });
    }
}
