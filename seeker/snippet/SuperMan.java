//date: 2022-04-15T17:05:34Z
//url: https://api.github.com/gists/6f007fc157a32c8ca70159c31c144985
//owner: https://api.github.com/users/FahimFBA

class Man{
    private int weight;
    public Man(int weight){
        this.weight = weight;
    }
    public void setWeight(int x) // setter method
    {
        this.weight = x;
    }
    public int getWeight() // getter method
    {
        return weight;
    }
    public void display(){
        System.out.println("Weight is: " + weight);
    }
}

public class SuperMan extends Man{
    public int myWeight;
    public SuperMan(int myWeight){
        super(58); // invoking the parent class constructor
        this.myWeight = myWeight;
    }
    public SuperMan(int weight, int myWeight) // overloading
    {
        super(weight); // invoking the parent class constructor
        this.myWeight = myWeight;
    }
    public void display() // overriding method
    {
        super.display();
        System.out.println("My weight is: " + myWeight);
    }
    public void display(String myString) // overloading method
    {
        System.out.println(myString);
    }
    public static void main(String[] args){
        SuperMan superObject = new SuperMan(25);
        superObject.display();
        SuperMan anotherObject = new SuperMan(55);
        anotherObject.display("I am a string value");
    }
}