//date: 2023-10-23T17:00:22Z
//url: https://api.github.com/gists/0840dc3dc4518567b83967e08f95a3ae
//owner: https://api.github.com/users/AaronRubinos

package AdapterPattern;

public class SmartphoneAdapter implements PowerOutlet {
    private SmartphoneCharger charger;

    public SmartphoneAdapter(SmartphoneCharger charger) {
        this.charger = charger;
    }

    @Override
    public void plugIn() {
        charger.chargePhone();
    }
}
