//date: 2022-05-10T17:18:34Z
//url: https://api.github.com/gists/ccb11eda8789ac029db921b99d5f5a5d
//owner: https://api.github.com/users/Dani-Sotano

package tictactoe.userinterface;

import java.util.Scanner;

public class ScannerService implements UserInterface {
    private final Scanner scanner = new Scanner(System.in);

    public ScannerService(){
    }

    public String getNextInput(){
        return scanner.nextLine();
    }

    public void nextOutput(String nextOutput){
        System.out.println(nextOutput);
    }
}