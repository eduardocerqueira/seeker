//date: 2022-02-10T16:43:40Z
//url: https://api.github.com/gists/ee646da16066f5e1392741b83f524e70
//owner: https://api.github.com/users/lismsdh2

public class Q1 {

    public static void main(String[] args) {
        
        String name = "도현호";
        char startValue = Character.MIN_VALUE;
        char endValue = Character.MAX_VALUE;
        char koStart = '\uAC00';
        char koEnd = '\uD7Af';


        
        for(int i = 0 ; i < name.length() ; i++){
            for(int j = startValue ; j < endValue ; j++){
                if(j >= (int)koStart && j <= (int)koEnd){
                    char c1 = name.charAt(i);
                    if(c1 == j){
                        System.out.printf("0x%s",Integer.toHexString(j));
                        if(i < 2){
                            System.out.print(", ");
                        }
                    }
                }
            }
        }
    
    }
    
}