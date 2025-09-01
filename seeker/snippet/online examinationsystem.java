//date: 2025-09-01T17:08:28Z
//url: https://api.github.com/gists/45e7fc282763a8a00a7eeb4e314aaf23
//owner: https://api.github.com/users/Nissssssshaa

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Timer;
import java.util.TimerTask;

class Question {
    String questionText;
    List<String> options;
    int correctOption;

    public Question(String questionText, List<String> options, int correctOption) {
        this.questionText = questionText;
        this.options = options;
        this.correctOption = correctOption;
    }
}

class User {
    String username;
    String password;

    public User(String username, String password) {
        this.username = username;
        this.password = "**********"
    }
}

public class OnlineExaminationSystem {

    static List<Question> questions = new ArrayList<>();
    static List<User> users = new ArrayList<>();
    static User loggedInUser;
    static boolean sessionOpen = false;
    static int score = 0;
    static Timer timer;
    static final int EXAM_DURATION_SECONDS = 60; // 1 minute timer

    public static void main(String[] args) {

        // Add a default admin user for initial testing
        users.add(new User("admin", "admin123"));
        initializeQuestions();
        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println("\n--- Online Examination System ---");
            System.out.println("1. Login");
            System.out.println("2. Register");
            System.out.println("3. Exit");
            System.out.print("\nEnter your choice: ");
            int choice = -1;
            try {
                choice = scanner.nextInt();
            } catch (java.util.InputMismatchException e) {
                System.out.println("Invalid input. Please enter a number.");
                scanner.nextLine(); // Consume the invalid input
                continue;
            }
            scanner.nextLine(); // Consume the newline character

            switch (choice) {
                case 1:
                    login(scanner);
                    break;
                case 2:
                    register(scanner);
                    break;
                case 3:
                    System.out.println("Exiting the system.");
                    System.exit(0);
                default:
                    System.out.println("Invalid choice. Please select a valid option!");
            }
        }
    }
    
    //-----------------------------------------------------------------------------------------------------------------------------------

    public static void initializeQuestions() {
        List<String> options1 = List.of("Guido van Rossum", "James Gosling", "Dennis Ritchie", "Bjarne Stroustrup");
        List<String> options2 = List.of("JRE", "JIT", "JDK", "JVM");
        List<String> options3 = List.of("Object-oriented", "Use of pointers", "Portable", "Dynamic and Extensible");
        List<String> options4 = List.of(".js", ".txt", ".class", ".java");
        List<String> options5 = List.of("Polymorphism", "Inheritance", "Compilation", "Encapsulation");
        
        questions.add(new Question("Who invented Java Programming?", options1, 1));
        questions.add(new Question("Which component is used to compile, debug and execute the java programs?", options2, 2));
        questions.add(new Question("Which one of the following is not a Java feature?", options3, 1));
        questions.add(new Question("What is the extension of Java code files?", options4, 3));
        questions.add(new Question("Which of the following is not an OOPS concept in Java?", options5, 2));
    }
    
    //-----------------------------------------------------------------------------------------------------------------------------------

    public static void register(Scanner scanner) {
        System.out.print("Enter a Username: ");
        String username = scanner.nextLine();
        System.out.print("Enter a Password: "**********"
        String password = "**********"

        for (User user : users) {
            if (user.username.equalsIgnoreCase(username)) {
                System.out.println("Username already taken. Please choose another one.");
                return;
            }
        }
        User newUser = "**********"
        users.add(newUser);
        System.out.println("Registration successful! You can now log in.");
    }
    
    //-----------------------------------------------------------------------------------------------------------------------------------

    public static void login(Scanner scanner) {
        System.out.print("Enter username: ");
        String username = scanner.nextLine();
        System.out.print("Enter password: "**********"
        String password = "**********"

        for (User user : users) {
            if (user.username.equals(username) && user.password.equals(password)) {
                loggedInUser = user;
                sessionOpen = true;
                startExam(scanner);
                return;
            }
        }
        System.out.println("Invalid credentials. Please try again!");
    }
    
    //-----------------------------------------------------------------------------------------------------------------------------------

    public static void startExam(Scanner scanner) {
        System.out.println("\nWelcome, " + loggedInUser.username + "!");
        score = 0;

        timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (sessionOpen) {
                    System.out.println("\nTime's up! Submitting your exam automatically.");
                    submitExam();
                }
            }
        }, EXAM_DURATION_SECONDS * 1000);

        for (int i = 0; i < questions.size(); i++) {
            if (!sessionOpen) {
                break;
            }
            Question question = questions.get(i);
            System.out.println("\nQuestion " + (i + 1) + ": " + question.questionText);
            for (int j = 0; j < question.options.size(); j++) {
                System.out.println((j + 1) + ". " + question.options.get(j));
            }

            System.out.print("Select your answer (1-" + question.options.size() + "): ");
            int userChoice = -1;
            try {
                userChoice = scanner.nextInt();
            } catch (java.util.InputMismatchException e) {
                System.out.println("Invalid input. Skipping this question.");
                scanner.nextLine();
                continue;
            }
            scanner.nextLine();

            // The main fix: Compare the user's 1-based choice with the 0-based correct option
            if (userChoice - 1 == question.correctOption) {
                System.out.println("Correct! ✅");
                score++;
            } else {
                System.out.println("Incorrect! ❌");
            }
        }

        if (sessionOpen) {
            submitExam();
        }
    }
    
    //-----------------------------------------------------------------------------------------------------------------------------------

    public static void submitExam() {
        sessionOpen = false;
        if (timer != null) {
            timer.cancel();
        }
        System.out.println("\n--- Exam Completed ---");
        System.out.println("Your final score is: " + score + " out of " + questions.size());
        System.out.println("Thank you for using the Online Examination System!");
        loggedInUser = null;
    }
}l;
    }
}