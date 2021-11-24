//date: 2021-11-24T17:10:16Z
//url: https://api.github.com/gists/16484c67784177b6dc9cc75a8cfa4be3
//owner: https://api.github.com/users/BerkeSoysal

public class Main {

    public static void main(String[] args) {
        ShipmentParameters shipmentParameters = new ShipmentParameters();
        shipmentParameters.isSameCity = false;
        shipmentParameters.distance = 70;
        shipmentParameters.weight = 1;

        CargoFirm safeAndQuick = new SafeAndQuick();
        safeAndQuick.createShipment();
        safeAndQuick.displayStatus();
        safeAndQuick.performFastDelivery(shipmentParameters);
        safeAndQuick.performDiscount(shipmentParameters); //discount can't be performed, not same city.
        safeAndQuick.setDiscountBehavior(new DiscountByWeight());
        safeAndQuick.performDiscount(shipmentParameters); //discont can be applied based on weight.
        CargoFirm easyPeasy = new EasyPeasy();
        easyPeasy.createShipment();
        easyPeasy.displayStatus();
        easyPeasy.performDiscount(shipmentParameters); 
        easyPeasy.performFastDelivery(shipmentParameters); // in 1 day
        easyPeasy.setFastDeliveryBehavior(new FastDeliveryStrategyTwo()); //behavior changed
        easyPeasy.performFastDelivery(shipmentParameters); // in 2 days
    }
}