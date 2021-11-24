//date: 2021-11-24T16:52:48Z
//url: https://api.github.com/gists/767c7f1c105de462dc6b9f99cf2711f9
//owner: https://api.github.com/users/BerkeSoysal

public class EasyPeasy extends CargoFirm
{
    public EasyPeasy()
    {
        this.discountBehavior = new DiscountByWeight();
        this.fastDeliveryBehavior = new FastDeliveryStrategyOne();
    }

    @Override
    public void createShipment()
    {
        System.out.println("Creating shipment for EasyPeasy");
    }
}