//date: 2023-12-01T16:58:06Z
//url: https://api.github.com/gists/be28ceb8358334ed15f3153d2e160301
//owner: https://api.github.com/users/deanMoisakos

package lab5;
/**
*
* @author DEAN
*/
public class Lab5 {
      /**
    * @param args the command line arguments
    */
    public static void main(String[] args) {
    Rectangle rectangle1 = new Rectangle();
    System.out.println("The width of the first Rectangle is "+rectangle1.width);
    System.out.println("The height of the first Rectangle is "+rectangle1.height);
    System.out.println("The area of the first Rectangle is "+rectangle1.getArea());
    System.out.println("The area of the first Rectangle is "+rectangle1.getPerim());
    Rectangle rectangle2 = new Rectangle(5.5, 38.9);
    System.out.println("\nThe width of the second Rectangle is "+rectangle2.width);
    System.out.println("The height of the second Rectangle is "+rectangle2.height);
    System.out.println("The area of the second Rectangle is "+rectangle2.getArea());
    System.out.println("The area of the second Rectangle is "+rectangle2.getPerim());
  }
}

class Rectangle{
  double width, height;
  Rectangle(){
  width = 5.0;
  height = 60.0;
}
  
Rectangle(double newWidth, double newHeight){
  width = newWidth;
  height = newHeight;
}
  
double getArea(){
  return width*height;
}
double getPerim(){
  return ((width*2)+(height*2));
}
  
void setWidthHeight(double newWidth, double newHeight){
  width = newWidth;
  height = newHeight;
}
}