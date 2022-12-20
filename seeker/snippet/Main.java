//date: 2022-12-20T17:09:35Z
//url: https://api.github.com/gists/24419875fc036d26c4cb8d055245831b
//owner: https://api.github.com/users/Mikaeryu

public class Main {
    public static void main(String[] args) {
        //tests
        booleanExpression(true,true,false,false);
        booleanExpression(true,true,true,true);
        booleanExpression(false,false,false,false);
        booleanExpression(false,false,true,true);
        booleanExpression(true,true,false,false);
    }

    public static boolean booleanExpression(boolean a, boolean b, boolean c, boolean d) {

        boolean[] boolList = {a,b,c,d};
        int trueCount = 0;

        for (boolean bool: boolList) {
            if (bool == true) {
                trueCount++;
            }
        }

        boolean isTrue = false;
        if (trueCount == 2) {
            isTrue = true;
        }

        System.out.println(isTrue);
        return isTrue;
    } //end method
}