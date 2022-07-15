//date: 2022-07-15T16:54:44Z
//url: https://api.github.com/gists/f9f3a9b829e96b8d4b8e7eca75e02193
//owner: https://api.github.com/users/The-General-T

import java.util.Scanner;
import Utils.Chats;
import Utils.Colors;
import Utils.GameVars;

public class RPGGame {
    static void Intro() throws InterruptedException {
        System.out.println(RED+"********* Welcome to the RPG Game *********");
        Thread.sleep(1500);
        System.out.println(Prompt+"What is your name?"+RESET);
        Name = UI.nextLine();
        Thread.sleep(2000);
        System.out.format(Narrator+"Hello, "+YELLOW+"%s"+CYAN+".\n",Name);
        Thread.sleep(2000);
        System.out.println(Narrator+"You are a Kinight in the village of Dragon's Cove.");
        Thread.sleep(2000);
        System.out.println(Narrator+"A Courier approaches you. He looks like he is in a hurry.");
        Thread.sleep(2000);
        System.out.println(Courier+"I have a letter! Your hands only!");
        Thread.sleep(2000);
        System.out.println(Narrator+"You open the letter: You have been summoned by the King.");
        Thread.sleep(2000);
        System.out.println(Courier+"The King sounded furious when he sent me. You should go quickly!");
        Thread.sleep(200);
        System.out.println(Narrator+"You make haste to the castle.");
        Thread.sleep(200);
        System.out.println(King+"Approach, knight.");

    }
    static void Start() throws InterruptedException {

    }

    public static void main(String[] args) {

    }
}
