//date: 2023-05-05T17:08:46Z
//url: https://api.github.com/gists/9592131b57c831576a6ef5c856697f3e
//owner: https://api.github.com/users/frankxhunter

package Logic;

public class TreeNode {
	
	private TreeNode leftNode;
	
	private TreeNode rightNode;
	
	private char data;
	
	public TreeNode(TreeNode leftNode, TreeNode rightNode, char data) {
		setLeftNode(leftNode);
		setRightNode(rightNode);
		setData(data );
	}
	
	public char getData() {
		return data;
	}

	public void setData(char data) {
		this.data = data;
	}

	public TreeNode getRightNode() {
		return rightNode;
	}

	public void setRightNode(TreeNode rightNode) {
		this.rightNode = rightNode;
	}

	public TreeNode getLeftNode() {
		return leftNode;
	}

	public void setLeftNode(TreeNode leftNode) {
		this.leftNode = leftNode;
	}


}
