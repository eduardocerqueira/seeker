//date: 2023-05-05T17:08:46Z
//url: https://api.github.com/gists/9592131b57c831576a6ef5c856697f3e
//owner: https://api.github.com/users/frankxhunter

package Logic;

import java.util.ArrayList;

public class testTree {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		TreeNode nodeA = new TreeNode(null, null, 'a');
		TreeNode nodeR = new TreeNode(null, null, 'r');
		TreeNode node1 = new TreeNode(nodeA, nodeR, ' ');
		TreeNode nodeE = new TreeNode(null, null, 'e');
		TreeNode node2 = new TreeNode(node1, nodeE, ' ');
		TreeNode nodeD = new TreeNode(null, null, 'd');
		TreeNode node3 = new TreeNode(node2, nodeD, ' ');
		TreeNode nodeS = new TreeNode(null, null, 's');
		TreeNode node4 = new TreeNode(node3, nodeS, ' ');
		TreeNode nodeO = new TreeNode(null, null, 'o');
		TreeNode node5 = new TreeNode(node4, nodeO, ' ');
		TreeNode nodeL = new TreeNode(null, null, 'l');
		TreeNode node6 = new TreeNode(node5, nodeL, ' ');
		
		System.out.println(testTree.decodeWord("reso", node6));


	}
	
	public static ArrayList<String> decodeWord(String word, TreeNode rootNode) {
		
		ArrayList<String> codedWord = new ArrayList<>();
		
		ArrayList<String> auxList = new ArrayList<>();
		
		TreeNode currentNode = rootNode;
		
		int treeCounter = 0;
		int charCounter = 0;
		int secondCounter = 0;
		char currentChar = ' ';
		boolean wordCoded = false;
		
		while(!wordCoded) {
			currentChar = word.charAt(charCounter);
			currentNode = rootNode;
			secondCounter = 0;
			
		while(currentNode.getRightNode() != null) {
			if(secondCounter == treeCounter) {
				currentNode = currentNode.getRightNode();
				auxList.add("1");
			}else {
				currentNode = currentNode.getLeftNode();
				auxList.add("0");
				secondCounter++;
			}
		}
		
		if(currentNode.getData() == currentChar) {
			charCounter++;
			treeCounter = 0;
			codedWord.addAll(auxList);
		}else {
			treeCounter++;
			auxList.clear();
			
		}
		
		if(charCounter  >= word.length()) {
			wordCoded = true;
		}
		
		}
		return codedWord;
	}

}
