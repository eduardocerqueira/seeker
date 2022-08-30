//date: 2022-08-30T16:49:31Z
//url: https://api.github.com/gists/f60d1467a4fd69545ce448851a82db41
//owner: https://api.github.com/users/halitgorgulu

public class IndentExample  
{  
    public static void main(String args[])   
    {  
           String text = "     Hello\n          from\n    otherside";  
           //değişmeden kalır
           System.out.println(text.indent(0));  
           //n tane boşluk silinir  
           System.out.println(text.indent(-6));  
           //n tane boşluk eklenir  
           System.out.println(text.indent(9));  
    }  
}
/*
     Hello
          from
    otherside

Hello
    from
otherside

              Hello
                   from
             otherside

*/