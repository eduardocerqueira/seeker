//date: 2023-01-09T17:00:16Z
//url: https://api.github.com/gists/88f6033f552b9b8bcbd274b92f4f3722
//owner: https://api.github.com/users/ILoveBacteria

class BST_Tree {
    Node nodeHead;
    
    /**
     * @param customRoot The custom node is a root that will be calculate number of sub trees from that
     */
    int countSubTree(Node customRoot) {
        if (customRoot.left != null && customRoot.right != null) {
            int subTreeLeft = countSubTree(customRoot.left);
            int subTreeRight = countSubTree(customRoot.right);
            return (subTreeLeft * subTreeRight) + (subTreeLeft + subTreeRight) + 1;
        }
        if (customRoot.left != null) {
            int subTreeLeft = countSubTree(customRoot.left);
            return subTreeLeft + 1;
        }
        if (customRoot.right != null) {
            int subTreeRight = countSubTree(customRoot.right);
            return subTreeRight + 1;
        }
        return 1;
    }
}

class Node {
    int data;
    Node left;
    Node right;
}