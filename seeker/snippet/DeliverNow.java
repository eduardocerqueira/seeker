//date: 2021-11-24T17:02:09Z
//url: https://api.github.com/gists/7b99bd911a0933048bf0c2d5472d2ef3
//owner: https://api.github.com/users/BerkeSoysal

public class DeliverNow extends CargoFirm
{
    public DeliverNow()
    {
        this.discountBehavior = new DiscountByWeight();
        this.fastDeliveryBehavior = new FastDeliveryStrategyOne();
    }

    @Override
    public void createShipment()
    {
        System.out.println("Created a shipment for DeliverNow...");
    }
}