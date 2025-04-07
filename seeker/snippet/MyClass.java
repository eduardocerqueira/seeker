//date: 2025-04-07T17:13:09Z
//url: https://api.github.com/gists/fecbc272fe5418c01031f1ce710e8a23
//owner: https://api.github.com/users/DanTheEpicMan

public class MyClass {
    public static void main(String args[]) {
        RobotComp robotComp = new RobotComp();
        robotComp.AssignToMatches();

        robotComp.PrintOut();

        System.out.println("\nWinning Team: " + robotComp.GetWinner());
        System.out.println("Lowest Scoring Round: " + robotComp.GetLowestRound());
        System.out.println("Highest Scoring Round: " + robotComp.GetHighestRound());
        System.out.println("Judge for Highest Scoring Round: " + robotComp.GetHighJudge());
        System.out.println("Judge for Lowest Score in Highest Scoring Round: " + robotComp.GetLowJudge());
    }
}
