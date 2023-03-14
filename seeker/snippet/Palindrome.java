//date: 2023-03-14T16:56:33Z
//url: https://api.github.com/gists/8d29614b0be985d4904f7d07014b1109
//owner: https://api.github.com/users/mahesh504

Class Palindrome{
    public static void main(String[] args)  {
        System.out.println("Enter a number");
        Scanner scanner = new Scanner(System.in);
        int digit,result=0,input, num = scanner.nextInt();
        input = num;
        while(num != 0){
            digit = num % 10;
            result = result * 10 + digit;
            num = num / 10 ;
        }

        System.out.println(result == input ? "Its palindrom": "its not a palindrom");
    }
}