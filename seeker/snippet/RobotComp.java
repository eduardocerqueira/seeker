//date: 2025-04-07T17:13:09Z
//url: https://api.github.com/gists/fecbc272fe5418c01031f1ce710e8a23
//owner: https://api.github.com/users/DanTheEpicMan

import java.util.ArrayList;
import java.lang.Math;

public class RobotComp {
    private int[][] mat = new int[5][6];
    private String[][] jud = new String[5][6];
    private ArrayList<String> judges = new ArrayList<String>();

    public RobotComp() {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                mat[i][j] = (int) (Math.random() * 10 + 40);
            }
        }

        judges.add("Smith");
        judges.add("Johnson");
        judges.add("Jones");
        judges.add("Patel");
        judges.add("Brandt");
        judges.add("Williams");
    }

    public void AssignToMatches() {
        for (int col = 0; col < 6; col++) {
            ArrayList<String> juds = new ArrayList<String>(judges);
            for (int row = 0; row < 5; row++) {
                int index = (int) (Math.random() * juds.size());
                jud[row][col] = juds.get(index);
                juds.remove(index);
            }
        }
    }

    public String GetWinner() {
        int[] teamTotals = new int[5];
        for (int i = 0; i < 5; i++) {
            int maxScore = 0;
            for (int j = 0; j < 6; j++) {
                if (mat[i][j] > maxScore) {
                    maxScore = mat[i][j];
                }
            }
            teamTotals[i] = maxScore;
        }

        int winner = 0;
        for (int i = 1; i < 5; i++) {
            if (teamTotals[i] > teamTotals[winner]) {
                winner = i;
            }
        }

        return "Team " + (winner + 1);
    }

    public int GetLowestRound() {
        int lowest = Integer.MAX_VALUE;
        int round = -1;
        for (int i = 0; i < 6; i++) {
            int roundTotal = 0;
            for (int j = 0; j < 5; j++) {
                roundTotal += mat[j][i];
            }
            if (roundTotal < lowest) {
                lowest = roundTotal;
                round = i;
            }
        }
        return round + 1;
    }

    public int GetHighestRound() {
        int highest = Integer.MIN_VALUE;
        int round = -1;
        for (int i = 0; i < 6; i++) {
            int roundTotal = 0;
            for (int j = 0; j < 5; j++) {
                roundTotal += mat[j][i];
            }
            if (roundTotal > highest) {
                highest = roundTotal;
                round = i;
            }
        }
        return round + 1;
    }

    public String GetHighJudge() {
        int highestRound = GetHighestRound() - 1;
        int highestScoreRow = 0;
        for (int i = 1; i < 5; i++) {
            if (mat[i][highestRound] > mat[highestScoreRow][highestRound]) {
                highestScoreRow = i;
            }
        }
        return jud[highestScoreRow][highestRound];
    }

    public String GetLowJudge() {
        int highestRound = GetHighestRound() - 1;
        int lowestScoreRow = 0;
        for (int i = 1; i < 5; i++) {
            if (mat[i][highestRound] < mat[lowestScoreRow][highestRound]) {
                lowestScoreRow = i;
            }
        }
        return jud[lowestScoreRow][highestRound];
    }

    public String GetJudgeFromRound(int a, int b) {
        return jud[a - 1][b - 1];
    }

    public void PrintOut() {
        System.out.print("Scores Array:");
        for (int[] row : mat) {
            System.out.println("");
            for (int x : row)
                System.out.print(x + " ");
        }

        System.out.println("\n");
        System.out.print("Judges Array:");
        for (String[] row : jud) {
            System.out.println("");
            for (String x : row) {
                System.out.print(x + " ");
            }
        }
        System.out.println("\n");
    }
}
