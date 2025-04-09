//date: 2025-04-09T17:13:04Z
//url: https://api.github.com/gists/7be7fab6e6e31b4fcddddb1d8f625f2a
//owner: https://api.github.com/users/thinkphp

public class Dreptunghi extends FormaGeometrica {

             private double lungime;
             private double latime;

             public Dreptunghi(String culoare, double lungime, double latime) {

                    super(culoare);
                    this.lungime = lungime;
                    this.latime = latime;
             }

             @Override
             public double calculeazaArea() {
                     return lungime * latime;
             }

             @Override
             public double calculeazaPerimetru() {
               return 2 * (lungime + latime);
             }
                       

}
