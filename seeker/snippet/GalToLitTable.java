//date: 2022-12-15T16:42:02Z
//url: https://api.github.com/gists/ec347c0f72b6cb98ff1e1d0a97db22b9
//owner: https://api.github.com/users/ivan-32682

public class GalToLitTable {
    public static void main(String []args) {
        double gallons, liters;
        int counter;

        counter=0;
        for(gallons=1; gallons<=100; gallons++) {
            liters=gallons*3.7854;
            System.out.println(gallons + " галлонам соответствует " + liters + " литра.");

            counter++;
            if(counter==10) {
                System.out.println();
                counter = 0;
            }
        }
    }
}
