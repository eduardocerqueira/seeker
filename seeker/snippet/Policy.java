//date: 2025-10-28T17:04:46Z
//url: https://api.github.com/gists/d695ad5331627d23981cf4db384ecca6
//owner: https://api.github.com/users/metalgamergod

/**
 * Policy.java
 * Models an insurance policy.
 */

public class Policy {
    // Attributes
    private int policyNumber;
    private String providerName;
    private String firstName;
    private String lastName;
    private int age;
    private String smokingStatus;  // "smoker" or "non-smoker"
    private double height; // in inches
    private double weight; // in pounds

    // No-arg constructor
    public Policy() {
        policyNumber = 0;
        providerName = "";
        firstName = "";
        lastName = "";
        age = 0;
        smokingStatus = "non-smoker";
        height = 0.0;
        weight = 0.0;
    }

    // Constructor with arguments
    public Policy(int policyNumber, String providerName, String firstName, String lastName,
                  int age, String smokingStatus, double height, double weight) {
        this.policyNumber = policyNumber;
        this.providerName = providerName;
        this.firstName = firstName;
        this.lastName = lastName;
        this.age = age;
        this.smokingStatus = smokingStatus;
        this.height = height;
        this.weight = weight;
    }

    // Accessors (Getters)
    public int getPolicyNumber() { return policyNumber; }
    public String getProviderName() { return providerName; }
    public String getFirstName() { return firstName; }
    public String getLastName() { return lastName; }
    public int getAge() { return age; }
    public String getSmokingStatus() { return smokingStatus; }
    public double getHeight() { return height; }
    public double getWeight() { return weight; }

    // Mutators (Setters)
    public void setPolicyNumber(int policyNumber) { this.policyNumber = policyNumber; }
    public void setProviderName(String providerName) { this.providerName = providerName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }
    public void setLastName(String lastName) { this.lastName = lastName; }
    public void setAge(int age) { this.age = age; }
    public void setSmokingStatus(String smokingStatus) { this.smokingStatus = smokingStatus; }
    public void setHeight(double height) { this.height = height; }
    public void setWeight(double weight) { this.weight = weight; }

    // Method to calculate BMI
    public double calculateBMI() {
        return (weight * 703) / (height * height);
    }

    // Method to calculate Policy Price
    public double calculatePolicyPrice() {
        double baseFee = 600.0;
        double additionalFee = 0.0;

        if (age > 50)
            additionalFee += 75.0;

        if (smokingStatus.equalsIgnoreCase("smoker"))
            additionalFee += 100.0;

        double bmi = calculateBMI();
        if (bmi > 35)
            additionalFee += (bmi - 35) * 20;

        return baseFee + additionalFee;
    }
}
