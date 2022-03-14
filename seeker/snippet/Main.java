//date: 2022-03-14T16:49:45Z
//url: https://api.github.com/gists/bec26a7494091ad6bc30e5a77c619743
//owner: https://api.github.com/users/dkkasturia

package Stellar;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


public class Main {
	public static int calculateRewardPoints(String [] purchases) {
		int totalRewardPoints =0;
		for (int i=1; i <purchases.length; i++) {
			int rewardPoints=0;
			int purchaseAmt = Integer.parseInt(purchases[i]);
			if (purchaseAmt > 100) {
				rewardPoints= (purchaseAmt - 100)*2 +50;
			} else if (purchaseAmt > 50) {
				rewardPoints = purchaseAmt - 50;
			}
			totalRewardPoints+= rewardPoints;			
		}
		return totalRewardPoints;
		
	}
	public static void main(String [] args) {
		String fileName = ".\\src\\purchases.txt";
		int lineNr=0;
		File file = new File(fileName);
		
		try {
			Scanner inputStream = new Scanner(file);
			while (inputStream.hasNextLine()) {
				lineNr++;
				String data = inputStream.nextLine();
				System.out.println(data);
				if (lineNr == 1) continue;
				String[] values = data.split("\t");
				int rewardPts = calculateRewardPoints(values);
				System.out.println("Reward points for the month " + values[0] + ":" + rewardPts);
			}
			inputStream.close();
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	  
}