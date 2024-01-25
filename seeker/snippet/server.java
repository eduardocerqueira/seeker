//date: 2024-01-25T17:06:00Z
//url: https://api.github.com/gists/a6f01d32d141aac285c9f7619b1b091c
//owner: https://api.github.com/users/Viz-Ar

package chat.application;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.text.*;
import java.util.*;
import java.net.ServerSocket;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;

public class server  implements ActionListener {
    JTextField text1;
    JPanel t1;
    static DataOutputStream dout ;
    static Box vertical = Box.createVerticalBox();
    static JFrame f = new JFrame();

    server() {
        f.setLayout(null);

        JPanel A1 = new JPanel();
        A1.setBackground(new Color(7, 94, 84));
        A1.setBounds(0, 0, 450, 60);
        A1.setLayout(null);
        f.add(A1);

        ImageIcon i1= new ImageIcon(ClassLoader.getSystemResource("icons/3.png"));
        Image i2 = i1.getImage().getScaledInstance(25, 25, Image.SCALE_DEFAULT);
        ImageIcon i3 = new ImageIcon(i2);
        JLabel back = new JLabel(i3);
        back.setBounds(5, 20, 25, 25);
        A1.add(back);



        back.addMouseListener(new MouseAdapter(){
            public void mouseClicked(MouseEvent ae){
                System.exit(0);}
        });

        ImageIcon i4= new ImageIcon(ClassLoader.getSystemResource("icons/rabina.jpeg"));
        Image i5 = i4.getImage().getScaledInstance(30, 30, Image.SCALE_DEFAULT);
        ImageIcon i6 = new ImageIcon(i5);
        JLabel profile = new JLabel(i6);
        profile.setBounds(40, 10, 50, 50);
        A1.add(profile);

        ImageIcon i7= new ImageIcon(ClassLoader.getSystemResource("icons/video.png"));
        Image i8 = i7.getImage().getScaledInstance(35, 30, Image.SCALE_DEFAULT);
        ImageIcon i9 = new ImageIcon(i8);
        JLabel video = new JLabel(i9);
        video.setBounds(310, 20, 30, 30);
        A1.add(video);

        ImageIcon i10= new ImageIcon(ClassLoader.getSystemResource("icons/phone.png"));
        Image i11 = i10.getImage().getScaledInstance(35, 30, Image.SCALE_DEFAULT);
        ImageIcon i12 = new ImageIcon(i11);
        JLabel phone = new JLabel(i12);
        phone.setBounds(360, 20, 35, 30);
        A1.add(phone);

        ImageIcon i13= new ImageIcon(ClassLoader.getSystemResource("icons/3icon.png"));
        Image i14 = i13.getImage().getScaledInstance(10, 25, Image.SCALE_DEFAULT);
        ImageIcon i15 = new ImageIcon(i14);
        JLabel more = new JLabel(i15);
        more.setBounds(400, 20, 30, 30);
        A1.add(more);


        JLabel name = new JLabel("Sira");
        name.setBounds(100, 15, 100,18);
        name.setForeground(Color.white);
        name.setFont(new Font("SAN_SERIF",Font.BOLD,18));
        A1.add(name);

        JLabel status = new JLabel("Active Now");
        status.setBounds(100, 30, 100,18);
        status.setForeground(Color.white);
        status.setFont(new Font("SAN_SERIF",Font.BOLD,14));
        A1.add(status);


        t1 = new JPanel();
        t1.setBounds(5, 75, 440, 570);
        f.add(t1);

        text1 = new JTextField();
        text1.setBounds(5, 655, 310, 40);
        text1.setFont(new Font("SAN_SERIF", Font.PLAIN, 16));
        f.add(text1);

        JButton send = new JButton("Send");
        send.setBounds(320, 655, 123, 40);
        send.setBackground(new Color(7, 94, 84));
        send.setForeground(Color.white);
        send.addActionListener(this);
        send.setFont(new Font("SAN_SERIF", Font.PLAIN, 16));
        f.add(send);

        f.  setSize(450, 700);
        f. setLocation(150, 50);
        f.  setUndecorated(true);
        f. getContentPane().setBackground(Color.white);
        f.   setVisible(true);
    }

    public void actionPerformed(ActionEvent ae) {
        try{
            String out = text1.getText();
            System.out.println(out);

            JPanel A2 = formatLabel(out);

            t1.setLayout(new BorderLayout());
            JPanel right = new JPanel(new BorderLayout());
            right.add(A2, BorderLayout.LINE_END);
            vertical.add(right);
            vertical.add(Box.createVerticalStrut(10));
            t1.add(vertical, BorderLayout.PAGE_START);

            dout.writeUTF(out);
            text1.setText("");


            f. repaint();
            f. invalidate();
            f. validate();
        }catch(Exception e){
            e.printStackTrace();}
    }

    public static JPanel formatLabel(String out) {
        JPanel panel = new JPanel();

        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        JLabel output = new JLabel("<html><p style=\"width:150px\">" + out + "</p></html>");
        output.setFont(new Font("Tahoma", Font.PLAIN, 16));
        output.setBackground(new Color(37, 211, 102));
        output.setOpaque(true);
        output.setBorder(new EmptyBorder(15, 15, 15, 50));

        panel.add(output);
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm");
        JLabel time = new JLabel(dateFormat.format(cal.getTime()));
        panel.add(time);

        return panel;
    }

    public static void main(String[] args) {
        new server();
        try{
            ServerSocket skt = new ServerSocket(6001);
            while(true){
                Socket s = skt.accept();
                DataInputStream din = new DataInputStream(s.getInputStream());
                dout = new DataOutputStream(s.getOutputStream());

                while (true){
                    String  msg = din.readUTF();
                    JPanel panel = formatLabel(msg);

                    JPanel left = new JPanel(new BorderLayout());
                    left.add(panel,BorderLayout.LINE_START);
                    vertical.add(left);
                    f.  validate();
                }
            }
        }catch(Exception e ) {
            e.printStackTrace();

        }
    }
}