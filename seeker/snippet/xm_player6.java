//date: 2023-06-27T16:52:04Z
//url: https://api.github.com/gists/288f3ba74310aa5369fb1e23d5387d77
//owner: https://api.github.com/users/EncodeTheCode

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.util.Base64;

public class XMPlayer extends JFrame {
    private JButton loadButton;
    private JLabel fileLabel;
    private JButton playButton;
    private JButton pauseButton;
    private JButton resumeButton;
    private JButton stopButton;
    private String loadedFile;
    private javax.sound.sampled.Clip clip;

    public XMPlayer() {
        setTitle("XM Player");
        setSize(400, 200);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        getContentPane().setBackground(Color.BLACK);
        setLayout(null);

        loadButton = new JButton("Load File");
        loadButton.setBounds(10, 10, 100, 30);
        loadButton.setForeground(new Color(102, 255, 102));
        loadButton.setBackground(Color.BLACK);
        loadButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loadFile();
            }
        });
        add(loadButton);

        fileLabel = new JLabel(" Filename: ");
        fileLabel.setBounds(10, 50, 380, 20);
        fileLabel.setForeground(new Color(102, 255, 102));
        fileLabel.setBackground(Color.BLACK);
        add(fileLabel);

        playButton = new JButton("Play");
        playButton.setBounds(10, 80, 100, 30);
        playButton.setForeground(new Color(102, 255, 102));
        playButton.setBackground(Color.BLACK);
        playButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                playMusic();
            }
        });
        add(playButton);

        pauseButton = new JButton("Pause");
        pauseButton.setBounds(10, 120, 100, 30);
        pauseButton.setForeground(new Color(102, 255, 102));
        pauseButton.setBackground(Color.BLACK);
        pauseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                pauseMusic();
            }
        });
        add(pauseButton);

        resumeButton = new JButton("Resume");
        resumeButton.setBounds(10, 160, 100, 30);
        resumeButton.setForeground(new Color(102, 255, 102));
        resumeButton.setBackground(Color.BLACK);
        resumeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                resumeMusic();
            }
        });
        add(resumeButton);

        stopButton = new JButton("Stop");
        stopButton.setBounds(120, 160, 100, 30);
        stopButton.setForeground(new Color(102, 255, 102));
        stopButton.setBackground(Color.BLACK);
        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                stopMusic();
            }
        });
        add(stopButton);

        setVisible(true);
    }

    private void loadFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.getName().toLowerCase().endsWith(".xm") || f.getName().toLowerCase().endsWith(".xm.b64") || f.isDirectory();
            }

            @Override
            public String getDescription() {
                return "XM Files (*.xm, *.xm.b64)";
            }
        });

        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            String filePath = file.getAbsolutePath();

            if (filePath.toLowerCase().endsWith(".xm.b64")) {
                try {
                    byte[] encodedData = Files.readAllBytes(file.toPath());
                    byte[] xmData = Base64.getDecoder().decode(encodedData);
                    String tempFilePath = filePath.substring(0, filePath.length() - 4);
                    File tempFile = new File(tempFilePath);
                    FileOutputStream outputStream = new FileOutputStream(tempFile);
                    outputStream.write(xmData);
                    outputStream.close();
                    filePath = tempFilePath;
                } catch (IOException e) {
                    JOptionPane.showMessageDialog(this, "Failed to decode base64 encoded XM file.", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
            }

            fileLabel.setText(" Filename: " + file.getName());
            loadedFile = filePath;
        }
    }

    private void playMusic() {
        try {
            File file = new File(loadedFile);
            if (file.exists()) {
                clip = javax.sound.sampled.AudioSystem.getClip();
                javax.sound.sampled.AudioInputStream inputStream = javax.sound.sampled.AudioSystem.getAudioInputStream(file);
                clip.open(inputStream);
                clip.start();
            }
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Failed to play XM file.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void pauseMusic() {
        if (clip != null && clip.isRunning()) {
            clip.stop();
        }
    }

    private void resumeMusic() {
        if (clip != null && !clip.isRunning()) {
            clip.start();
        }
    }

    private void stopMusic() {
        if (clip != null) {
            clip.stop();
            clip.close();
            clip = null;
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new XMPlayer();
            }
        });
    }
}
