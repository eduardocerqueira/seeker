//date: 2022-03-16T17:03:06Z
//url: https://api.github.com/gists/bfdbb796e01d2033b3fec0e9fad52977
//owner: https://api.github.com/users/Amrit-kaur01

class Solution {
    public int romanToInt(String s) {
        int num=0;
        char prev='\u0000';
        for(int i=0;i<s.length();i++)
        {
            char ch=s.charAt(i);
            switch(ch)
            {
                case 'I':num+=1; break;
                case 'V':if(prev=='I') num+=3;
                        else num+=5;
                        break;
                case 'X': if(prev=='I') num+=8;
                        else num+=10;
                        break;
                case 'L':if(prev=='X') num+=30;
                        else num+=50;
                        break;
                case 'C':if(prev=='X') num+=80;
                        else num+=100;
                        break;
                case 'D':if(prev=='C') num+=300;
                        else num+=500;
                        break;
                case 'M':if(prev=='C') num+=800;
                        else num+=1000;
                        break;
            }
            prev=ch;
        }
        return num;
    }
}