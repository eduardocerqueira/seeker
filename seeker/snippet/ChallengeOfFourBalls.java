//date: 2022-02-23T17:06:12Z
//url: https://api.github.com/gists/7d7e48a934df7eda19885217a830a916
//owner: https://api.github.com/users/jastripathy

public class ChallengeOfFourBalls extends PApplet {
    public int height=680;
    public int width=680;
    public static int x_axis=0;
    public Balls ball_one=new Balls(x_axis,50,1,this);
    public Balls ball_two=new Balls(x_axis,100,2,this);
    public Balls ball_three=new Balls(x_axis,150,3,this);
    public Balls ball_four=new Balls(x_axis,200,4,this);
    public static void main(String args[]){
        PApplet.main("ChallengeOfFourBalls",args);
    }
    @Override
    public void settings(){
        super.settings();
        size(height,width);
    }

    private void size(int height, int width) {
    }

    @Override
    public void setup(){
    }
    @Override
    public void draw(){
        ball_one.DrawBall();
        ball_two.DrawBall();
        ball_three.DrawBall();
        ball_four.DrawBall();
    }
}
class Balls {
    public  int x_axis;
    public  int y_axis;
    public  int speed;
    public  int diameter=40;
    public PApplet Papplet ;
    public Balls(int x_axis, int y_axis, int speed, PApplet Papplet){
        this.x_axis=x_axis;
        this.y_axis=y_axis;
        this.speed=speed;
        this.Papplet=Papplet;
    }
    public void DrawBall(){
        Papplet.ellipse(x_axis,y_axis,diameter,diameter);
        IncreaseSpeedofBall();
    }
    private void IncreaseSpeedofBall() {
        x_axis += speed;
    }
}