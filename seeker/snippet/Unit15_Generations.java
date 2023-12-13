//date: 2023-12-13T16:58:02Z
//url: https://api.github.com/gists/ad88c78ae6efa47eab1ec132f87f1f9c
//owner: https://api.github.com/users/tgibbons-css

public class Unit15_Generations {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int age;
        
        // Input the users age
        System.out.println("Enter your age: ");
        age = input.nextInt();        
        // Print out the age
        System.out.print("The person's age is ");
        System.out.println(age);
        
        // Determine if the person is old enough to vote
        checkCanVote(age);
        
        // Identify the generation
        //String generation = getGeneration(age);
        System.out.println("The person belongs to the " + getGeneration(age) + ".");

    } // end main
    
    private static String getGeneration(int age) {
        String generation;
        if (age >= 75) {
            generation = "Silent Generation";
        } else if (age >= 56) {
            generation = "Baby Boomers";
        } else if (age >= 41) {
            generation = "Gen X";
        } else if (age >= 26) {
            generation = "Millennials";
        } else if (age >= 12) {
            generation = "Gen Z";
        }else {
            generation = "Alpha";
        }
        return generation;
    }
    
    private static void checkCanVote(int age) {
        // Determine if the person is old enough to vote
        if (age>=18) {
            System.out.println("The person can vote.");
        } else {
            System.out.println("The person can not vote yet.");
        } 
    }

    
}  // end of class
