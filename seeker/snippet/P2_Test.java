//date: 2023-02-07T17:10:01Z
//url: https://api.github.com/gists/cf17a9708a18b9fec4e352d6bc096eb6
//owner: https://api.github.com/users/strahil35364

package P2;

public class Test {
    public static void main(String[] args) {
        TruckDriver truckDriver1 = new TruckDriver();
        truckDriver1.name = "Strahil";
        truckDriver1.age = 21;
        truckDriver1.adres = "Teteven";

        TruckDriver truckDriver2 = new TruckDriver();
        truckDriver2.name = "Valentin";
        truckDriver2.age = 21;
        truckDriver2.adres= "Sofia";

        TruckDriver truckDriver3 = new TruckDriver();
        truckDriver3.name = "Stoyan";
        truckDriver3.age = 55;
        truckDriver3.adres = "Stara Zagora";

        TruckDriver truckDriver4 = new TruckDriver();
        truckDriver4.name = "Aleksandar";
        truckDriver4.age = 23;
        truckDriver4.adres = "Sofia";

        TruckDriver truckDriver5 = new TruckDriver();
        truckDriver5.name = "Tanislav";
        truckDriver5.age = 23;
        truckDriver5.adres = "Sofia";

        TruckDriver [] group = new TruckDriver[5];

        group [0] = truckDriver1;
        group [1] = truckDriver2;
        group [2] = truckDriver3;
        group [3] = truckDriver4;
        group [4] = truckDriver5;

        int i;
        for (i = 0; i < group.length; i++){
            if (group[i].age == 21 ){
                System.out.println(group [i].name);
            }
        }

    }
}
