//date: 2024-01-01T16:53:48Z
//url: https://api.github.com/gists/0855778865e9b6609a15fcce7830921f
//owner: https://api.github.com/users/Interested-Person

import java.util.*;
class brackets{
    static void main(){
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter mathematical expression");
        String s=sc.next();
        
        for(int i=0; i<s.length(); i++){
            boolean continueToNextIteration=false;
            char ch=s.charAt(i);
            boolean lookingfor1=false, lookingfor2=false, lookingfor3=false;
            switch(ch){
                case '(':
                    lookingfor1=true;
                    break;
                case ')':
                    System.out.println("Bracket ) at " +i+ " was never opened");
                    System.exit(0);
                case '{':
                    lookingfor2=true;
                    break;
                case '}':
                    System.out.println("Bracket } at " +i+ "was never opened");
                    System.exit(0);
                case '[':
                    lookingfor3=true;
                    break;
                case ']':
                    System.out.println("Bracket ] at " +i+ " was never opened");
                    System.exit(0);
                default:
                    continueToNextIteration=true;
            }
            if(continueToNextIteration){
                continue;
            }
            boolean foundClosingBracket=false;
            for(int j=i+1; j<s.length(); j++){
                char ch2=s.charAt(j);
                if(lookingfor1 && ch2==')'){
                    s=removeFromString(s, i, j);
                    foundClosingBracket=true;
                    break;
                }
                else if(lookingfor2 && ch2=='}'){
                    s=removeFromString(s, i, j);
                    foundClosingBracket=true;
                    break;
                }
                else if(lookingfor3 && ch2==']'){
                    s=removeFromString(s, i, j);
                    foundClosingBracket=true;
                    break;
                }
            }
            if(!foundClosingBracket){
                System.out.println("Bracket at " +i+ " never closed");
                System.exit(0);
            }
        }
        
    }
    static String removeFromString(String s, int ind1, int ind2){
        s=s.substring(0,ind1)+"_"+s.substring(ind1+1, s.length()); //removing ind1
        s=s.substring(0,ind2)+"_"+s.substring(ind2+1, s.length()); //removing ind2
        return s;
    }
}