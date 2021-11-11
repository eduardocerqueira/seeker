//date: 2021-11-11T17:14:49Z
//url: https://api.github.com/gists/fc3314e157354dde1658289c4cce9f57
//owner: https://api.github.com/users/johanneslosch

class Email
{
    String str1;
    int i, length;
    char ch;
    boolean test = true;
    public void take(String str)
    {
        i = str.indexOf('@');
        if(i == -1)
        {
            test = false;
            System.err.println("Invalid Email Id (‘@’ symbol missing).");
        }
        i = str.indexOf(' ');
        if(i != -1)
        {
            test = false;
            System.err.println("Invalid Email Id (space is not allowed).");
        }

        i = str.indexOf('.');
        if(i == -1)
        {
            test = false;
            System.err.println("Invalid Email Id (‘.’ missing).");
        }
        str1 = str.substring(i + 1);
        length = str1.length();
        for(i = 0; i< length; i++)
        {
            ch=str1.charAt(i);
            if(Character.isDigit(ch))
                break;
        }
        if(i != length)
        {
            test = false;
            System.err.println("Invalid Email Id (Digit is not permitted in extension part).");
        }
        if(test)
            System.out.println("Email Id = " + str);
    }

    public static void main(String args[])
    {
        Email ob=new Email();
        ob.take("email@example.com");
    }
}