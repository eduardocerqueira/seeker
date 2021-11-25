//date: 2021-11-25T17:10:08Z
//url: https://api.github.com/gists/3083d418224d81f1f9c03fca960e2e8e
//owner: https://api.github.com/users/shlomiaflalo

import java.util.*;

    public class Randomly{
    public static void main(String[]args){

        /*
        This code presents your text randomly
         */
        
        Scanner sc=new Scanner(System.in);
        Random r=new Random();
        int random_location;
        ArrayList My_list=new ArrayList();

        System.out.println("Please " +
        "Write down your sentence, I'll show it to you \n" +
        "in a random way");
        String[]Separate_my_sentence=sc.nextLine().split(" ");


        for(int i=0;i<Separate_my_sentence.length;){
        random_location=r.nextInt(Separate_my_sentence.length);

        if(My_list.size()==Separate_my_sentence.length){
            break;
        }

        if(!My_list.contains(random_location)){
            My_list.add(random_location);
            System.out.print(Separate_my_sentence[random_location]+" ");
            i++;
        }
        }

    }
}
