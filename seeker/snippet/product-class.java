//date: 2024-06-19T17:02:14Z
//url: https://api.github.com/gists/20018da26fc1728184bff6dfd8976e77
//owner: https://api.github.com/users/docsallover

public class Product {
  private int id; // Unique product identifier (private for data protection)
  private String name;
  private double price;
  private int quantity;

  // Constructor to initialize product attributes
  public Product(int id, String name, double price, int quantity) {
    this.id = id;
    this.name = name;
    this.price = price;
    this.quantity = quantity;
  }

  // Method to display product information
  public void displayProductInfo() {
    System.out.println("Product ID: " + id);
    System.out.println("Name: " + name);
    System.out.println("Price: $" + price);
    System.out.println("Quantity in Stock: " + quantity);
  }
  
  // Getter and setter methods for quantity (optional for controlled modification)
  public int getQuantity() {
    return quantity;
  }

  public void setQuantity(int newQuantity) {
    this.quantity = newQuantity;
  }
}