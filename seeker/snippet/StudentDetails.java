//date: 2022-03-22T17:01:28Z
//url: https://api.github.com/gists/f0f250c549eb8faf63dc6c6a69df55d3
//owner: https://api.github.com/users/laxmi2001

package CAT2.Practice;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class InvalidAadhar extends Exception{
    public InvalidAadhar(String message){
        super(message);
    }
}

class StudentInfo{
    private String name;
    private String regno;
    private String phno;
    private String aadhar;
    private String passport;

    public StudentInfo(String name, String regno, String phno, String aadhar, String passport) {
        this.name = name;
        this.regno = regno;
        this.phno = phno;
        this.aadhar = aadhar;
        this.passport = passport;
    }

    public static boolean AadharChecker(String aadhar) throws InvalidAadhar{
        String regex = "^[2-9]{1}[0-9]{3}\\s[0-9]{4}\\s[0-9]{4}$";
        Pattern p = Pattern.compile(regex);
        if(aadhar == null){
            return false;
        }
        Matcher m = p.matcher(aadhar);
        if(m.matches() == true){
            return true;
        }
        else{
            throw new InvalidAadhar("Invalid aadhar number");
        }
    }

    public static boolean PassportChecker(String passport) {
        String regex = "^[A-Z]{1}[0-9]{7}$";
        Pattern p = Pattern.compile(regex);
        if (passport == null) {
            return false;
        }
        Matcher m = p.matcher(passport);
        return m.matches();
    }

    @Override
    public String toString() {
        return "StudentInfo{" +
                "name='" + name + '\'' +
                ", regno='" + regno + '\'' +
                ", phno='" + phno + '\'' +
                ", aadhar='" + aadhar + '\'' +
                ", passport='" + passport + '\'' +
                '}';
    }
}

public class StudentDetails {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String n1;
        String reg1;
        String mb1;
        String ad1;
        String pass1;
        System.out.print("Enter name - ");
        n1 = sc.nextLine();
        System.out.print("Enter Registration number - ");
        reg1 = sc.nextLine();
        System.out.print("Enter Phone number - ");
        mb1 = sc.nextLine();
        System.out.print("Enter Aadhar number - ");
        ad1 = sc.nextLine();
        System.out.print("Enter Passport ID - ");
        pass1 = sc.nextLine();

        StudentInfo std1 = new StudentInfo(n1, reg1, mb1, ad1, pass1);
        System.out.println(std1);
        try{
            System.out.println(StudentInfo.AadharChecker(ad1));
        }
        catch (InvalidAadhar e){
            System.out.println(e.toString());
        }

        System.out.println(StudentInfo.PassportChecker(pass1));
    }
}
