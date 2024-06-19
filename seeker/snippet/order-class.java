//date: 2024-06-19T17:02:14Z
//url: https://api.github.com/gists/20018da26fc1728184bff6dfd8976e77
//owner: https://api.github.com/users/docsallover

public class Order {
  private Product product; // Reference to the product object
  private int quantityOrdered;
  private double totalOrderValue;

  // Constructor to initialize order details
  public Order(Product product, int quantityOrdered) {
    this.product = product;
    this.quantityOrdered = quantityOrdered;
    this.totalOrderValue = product.getPrice() * quantityOrdered; // Calculate total value
  }

  // Method to display order information
  public void displayOrderInfo() {
    System.out.println("Order Details:");
    product.displayProductInfo();  // Reuse product's display method
    System.out.println("Quantity Ordered: " + quantityOrdered);
    System.out.println("Total Order Value: $" + totalOrderValue);
  }
}