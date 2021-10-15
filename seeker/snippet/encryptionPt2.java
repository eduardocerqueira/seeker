//date: 2021-10-15T17:12:27Z
//url: https://api.github.com/gists/2e11d4af835b8d9fc98bfd39161b717a
//owner: https://api.github.com/users/Sivar24

import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseListener;
import java.awt.event.MouseEvent;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.Font;
import java.util.*;
import java.io.*;
public class PasswordProgram {
    private JFrame frame;

    public PasswordProgram() {
        frame = new JFrame("Password Program");
        frame.setSize(800, 800);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setPreferredSize(frame.getSize());
        frame.add(new InnerProgram(frame.getSize()));
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String... argv) {
        new PasswordProgram();
    }

    public static class InnerProgram extends JPanel implements Runnable, MouseListener  {

        ArrayList<String> names = new ArrayList<String>();
        private Thread animator;
        Dimension d;
        int startX = 20;
        int startY = 20;
        String masterpass = "";
        String name = "";
        String acc_name = "";
        String user = "";
        String pass = "";
        String message = "";
        
        String screen1message = "";
        int field = 1;
        
        int prev = 0;
        boolean screen1 = true;
        boolean newAcc = true;
        ReadWrite rw = new ReadWrite();
       
        public InnerProgram (Dimension dimension) {
            setSize(dimension);
            setPreferredSize(dimension);
            addMouseListener(this);
            addKeyListener(new TAdapter());
            setFocusable(true);
            d = getSize();


            //for animating the screen - you won't need to edit
            if (animator == null) {
                animator = new Thread(this);
                animator.start();
            }
            setDoubleBuffered(true);
        } 
        
        @Override
        public void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D)g;

            g2.setColor(Color.black);
            Color co = new Color(54,10,42);
            g2.setColor(co);
            g2.fillRect(0, 0,(int)d.getWidth() , (int)d.getHeight());
            startX = 20;
            startY = 20;
            int sideL = 100;
            int sideW = 150;
            int count=1;
            if(screen1){
                g2.setFont(new Font("TimesRoman", Font.PLAIN, 20));
                
                g2.setColor(Color.red);
                g2.fillRect(250,500,100,50);
                g2.setColor(Color.black);
                g2.drawString("submit", 275,525);

                g2.setColor(Color.white);
    
                g2.drawString("Master Password: " + masterpass,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 200);
                
                g2.setColor(Color.red);
                
                g2.setFont(new Font("TimesRoman", Font.PLAIN, 30));
                
                g2.drawString(screen1message ,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 50);
                
            } else {
                if(newAcc){
                    g2.setFont(new Font("TimesRoman", Font.PLAIN, 20));
                    
                    g2.setColor(Color.red);
                    g2.fillRect(250,600,300,50);
                    g2.setColor(Color.black);
                    g2.drawString("Already have an account?", 275,625);
                    
                    g2.setColor(Color.red);
                    g2.fillRect(575,600,100,50);
                    g2.setColor(Color.black);
                    g2.drawString("submit", 600,625);
                    
                    
                    g2.setColor(Color.white);

                    g2.drawString("Enter an Account Name: " + name,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 200);
                    g2.drawString("Enter an Account Username: " + user,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 150);
                    g2.drawString("Enter an Account Password: " + pass,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 100);
                    
                    g2.setColor(Color.red);
                    
                    g2.setFont(new Font("TimesRoman", Font.PLAIN, 30));
                    
                    g2.drawString(message ,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 50);
                } else {
                    g2.setFont(new Font("TimesRoman", Font.PLAIN, 20));
                    
                    g2.setColor(Color.red);
                    g2.fillRect(250,500,100,50);
                    g2.setColor(Color.black);
                    g2.drawString("submit", 275,525);
    
                    g2.setColor(Color.white);
        
                    g2.drawString("Enter Account Name: " + acc_name,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 200);
                    
                    g2.setColor(Color.red);
                    
                    g2.setFont(new Font("TimesRoman", Font.PLAIN, 30));
                    
                    g2.drawString(message ,220,((int)d.getHeight()-(sideL*2) + sideL/2) - 50);
                    
                }

            }


        }

        public static int random (int a, int b){    
            int max=a;
            int min=b;
            int random=(int)(Math.random() * (max - min) + min);

            return random;
        }
        
        public String Encrypt(String phrase, int shift){
            int key = 0;
            String encrypted = "";
            for(int i = 0; i < phrase.length(); i++){
                char c = phrase.charAt(i);
                key = (int)c - 32;
                key = ((key + shift) % 94);
                key += 32;

                encrypted += Character.toString((char) key);
            }
            System.out.println(encrypted);
            masterpass = encrypted;
            return masterpass;
        }
        public String Decrypt(String phrase, int shift){
            int key = 0;
            String decrypted = "";
            for(int i = 0; i < phrase.length(); i++){
                char c = phrase.charAt(i);
                key = (int)c - 32;
                key = ((key - shift) % 94);
                key += 32;

                decrypted += Character.toString((char) key);
            }
            System.out.println(decrypted);
            masterpass = decrypted;
            return masterpass;
        }
        public void mousePressed(MouseEvent e) {
            int x = e.getX();
            int y = e.getY();
            /*
            if((x > 100 && x < 200) && (y > 100 && y < 200)){
                Encrypt(masterpass, 3);
                System.out.println("Encrypted");
            }
            if((x > 100 && x < 200) && (y > 200 && y < 300)){
                Decrypt(masterpass, 3);
                System.out.println("Decrypted");
            }
            */
            if(screen1){
                if((x > 250 && x < 350) && (y > 500 && y < 550)){
                    ArrayList<String> lines = new ArrayList<String>();
                    lines = rw.read("passkeep.txt");
                    String s = lines.get(0);
                    if(masterpass.equals(s)){
                        screen1 = false;
                    } else {
                        screen1message = "Incorrect password, try again.";
                    }
                }
            } else {
                if(newAcc){
                    if((x > 250 && x < 550) && (y > 600 && y < 650)){
                        newAcc = false;
                    }
                    if((x > 500 && x < 800) && (y > 600 && y < 650)){
                        rw.write(name + ": " + user + " , " + Encrypt(pass, 4));
                    }
                } else {
                    if((x > 250 && x < 350) && (y > 500 && y < 550)){
                        message = "";
                        ArrayList<String> lines = new ArrayList<String>();
                        lines = rw.read("passkeep.txt");
                        for(int i = 0; i < lines.size(); i++){
                            if(lines.get(i).indexOf(acc_name) != -1){
                                String temp = lines.get(i);
                                message = temp.substring(0, temp.indexOf(",")) + "," + Decrypt(temp.substring(temp.indexOf(",") + 1), 4);
                            }
                        }
                        if(message.equals("")){
                            message = "There is no account of this name.";
                        }
                    }

                    
                }
            }
            
            
            

        }
        public void mouseReleased(MouseEvent e) {
        }
        public void mouseEntered(MouseEvent e) {
        }

        public void mouseExited(MouseEvent e) {
        }

        public void mouseClicked(MouseEvent e) {
        }

        private class TAdapter extends KeyAdapter {

            public void keyReleased(KeyEvent e) {
                int keyr = e.getKeyCode();

            }

            public void keyPressed(KeyEvent e) {
                int key = e.getKeyCode();
                String c = KeyEvent.getKeyText(e.getKeyCode());
                c = Character.toString((char) key);
                
                
                if(screen1){
                    if(prev!=16){
                        c = c.toLowerCase();
                    }
                        
                    prev = key;
                    if(key==8)
                        masterpass = masterpass.substring(0,masterpass.length()-1);
                    if(key==10){
                        masterpass="";
                    }
                    
                  
                    
                    if(key!=8 && key!=10 && key!=16)
                        masterpass += c;
                        
                    System.out.println( key + " - " +  c);
                        
                } else {
                    if(prev!=16){
                        c = c.toLowerCase();
                    }
                        
                    prev = key;
                    if(newAcc){
                        if(e.getKeyCode()==40){
                            field++;
                            if(field > 3){
                                field = 1;
                            }
                        } else if (e.getKeyCode() == 38){
                            field--;
                            if(field < 1){
                                field = 3;
                            }
                        }
                        if(field == 1){
                            if(key==8)
                                name = name.substring(0,name.length()-1);
                                
                            if(key==10){
                                name = "";
                            }
                            
                            if(key!=8 && key!=10 && key!=16 && key!=40 && key!=38)
                                name += c;
                                
                            System.out.println( key + " - " +  c);
                             
                        } else if(field == 2){
                            if(key==8)
                                user = user.substring(0,user.length()-1);
                            if(key==10){
                                user ="";
                            }
                            
                          
                            
                            if(key!=8 && key!=10 && key!=16 && key!=40 && key!=38)
                                user += c;
                                
                            System.out.println( key + " - " +  c);
                             
                        } else if(field == 3){
                            if(key==8)
                                pass = pass.substring(0,pass.length()-1);
                            if(key==10){
                                pass="";
                            }
                            
                          
                            
                            if(key!=8 && key!=10 && key!=16 && key!=40 && key!=38)
                                pass += c;
                                
                            System.out.println( key + " - " +  c);
                             
                        }
                        prev = key;
                        if(key==8)
                            masterpass = masterpass.substring(0,masterpass.length()-1);
                        if(key==10){
                            masterpass="";
                        }
                        
                      
                        
                        if(key!=8 && key!=10 && key!=16)
                            masterpass += c;
                            
                        System.out.println( key + " - " +  c);
                            
                    } else {
                        if(prev!=16){
                            c = c.toLowerCase();
                        }
                            
                        prev = key;
                        if(key==8)
                            acc_name = acc_name.substring(0,acc_name.length()-1);
                        if(key==10){
                            acc_name="";
                        }
                        
                      
                        
                        if(key!=8 && key!=10 && key!=16)
                            acc_name += c;
                            
                        System.out.println( key + " - " +  c);
                    }

                }
                
                

                // message = "Key Pressed: " + e.getKeyCode();
            }
        }//end of adapter

        public void run() {
            long beforeTime, timeDiff, sleep;
            beforeTime = System.currentTimeMillis();
            int animationDelay = 37;
            long time = System.currentTimeMillis();
            while (true) {// infinite loop
                // spriteManager.update();
                repaint();
                try {
                    time += animationDelay;
                    Thread.sleep(Math.max(0, time - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    System.out.println(e);
                } // end catch
            } // end while loop
        }// end of run
        
    }//end of class
}