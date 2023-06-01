//date: 2023-06-01T17:03:49Z
//url: https://api.github.com/gists/37b211a8ce7911cc2b9fd41ce5e33ab7
//owner: https://api.github.com/users/kumarsumiit

//using geter and setter set its radius and height.
//and calculate surfacearea and volume of cylinder.
class Cylinder{
    private int radius;
    private int height;

    public void setRadius(int r){
        radius = r;
    }
    public void setHeight(int h){
        height = h;
    }
    public int getRadius(){
        return radius;
    }
    public int getHeight(){
        return height;
    }
    public double surfacearea(){
        return (2* Math.PI * radius * height) + (2 * Math.PI * radius * radius);
    }
    public double volume(){
        return Math.PI * radius * radius * height;
    }


}


public class oopspractis2 {
    public static void main(String[] args) {
        Cylinder flor1st = new Cylinder();
        flor1st.setHeight(12);
        flor1st.setRadius(9);
        System.out.println(flor1st.getHeight()+" "+ flor1st.getRadius());
        System.out.println(flor1st.surfacearea());
        System.out.println(flor1st.volume());


    }
}
