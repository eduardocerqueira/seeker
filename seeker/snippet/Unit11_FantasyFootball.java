//date: 2024-11-15T17:12:18Z
//url: https://api.github.com/gists/fc1a432bc0eb455e4d88f42ed478da28
//owner: https://api.github.com/users/tgibbons-css

import java.util.Random;

public class Unit11_FantasyFootball {

    // Create an array of strings to store the names of top NFL wide receivers
    static String[] wideReceivers = {
        "DeAndre Hopkins", "Stefon Diggs", "Tyreek Hill", "Davante Adams", "Calvin Ridley",
        "Justin Jefferson", "A.J. Brown", "D.K. Metcalf", "Keenan Allen", "Julio Jones",
        "Adam Thielen", "Mike Evans", "Allen Robinson", "Chris Godwin", "Cooper Kupp",
        "Amari Cooper", "Robert Woods", "Tyler Lockett", "CeeDee Lamb", "Terry McLaurin",
        "Kenny Golladay", "DJ Moore", "Michael Thomas", "Odell Beckham Jr.", "JuJu Smith-Schuster",
        "Robby Anderson", "Courtland Sutton", "Brandon Aiyuk", "Jarvis Landry", "Tyler Boyd",
        "Marquise Brown", "Chase Claypool", "DeVante Parker", "Mike Williams", "Brandin Cooks",
        "Darnell Mooney", "Deebo Samuel", "Antonio Brown", "Jaylen Waddle"
    };

    public static void main(String[] args) {

        int wideReceiverData[][] = generateFantasyData();
        // wideReceiverData[0][player] is the receptions
        // wideReceiverData[1][player] is the receivingYards
        // wideReceiverData[2][player] is the receivingTouchdowns
        int[] score = new int[wideReceivers.length];
        // add code here to give each wide receiver an score based on their receptions, receiving Yards and receivingTouchdowns
        for (int i = 0; i < wideReceivers.length; i++) {
            int receptions = wideReceiverData[0][i];
            int recYards = wideReceiverData[1][i];
            int recTDs = wideReceiverData[2][i];
            score[i] = receptions + (recYards/10) + (recTDs*30);
        }
        // Display the information for each wide receiver
        System.out.printf("%-25s  %-15s  %-15s  %-15s %-15s \n", "Wide Receiver", "REC", "REC YDS", "REC TD", "Score");
        for (int i = 0; i < wideReceivers.length; i++) {
            System.out.printf("%-25s  %-15d  %-15d  %-15d  %-15d  \n",
                    wideReceivers[i], wideReceiverData[0][i], wideReceiverData[1][i], wideReceiverData[2][i],score[i]);
        }
    }

    // This code generates some randome data for the wide receivers
    private static int[][] generateFantasyData() {
        // Create arrays for receptions (REC), receiving yards (REC YDS), and receiving touchdowns (REC TD)
        int[] receptions = new int[wideReceivers.length];
        int[] receivingYards = new int[wideReceivers.length];
        int[] receivingTouchdowns = new int[wideReceivers.length];

        // Initialize arrays with random values within appropriate ranges
        Random random = new Random();
        for (int i = 0; i < wideReceivers.length; i++) {
            receptions[i] = random.nextInt(100) + 40; // Random value between 40 and 139
            receivingYards[i] = random.nextInt(1000) + 400; // Random value between 400 and 1399
            receivingTouchdowns[i] = random.nextInt(10); // Random value between 0 and 9
        }

        int fantasyData[][] = {receptions, receivingYards, receivingTouchdowns};
        return fantasyData;
    }
   
}