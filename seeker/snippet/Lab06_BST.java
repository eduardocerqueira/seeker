//date: 2025-03-17T16:48:35Z
//url: https://api.github.com/gists/23b1f3ad16672a4c7f2e43101edcf1dc
//owner: https://api.github.com/users/gxjakkap

// 67070501056 Jakkaphat Chalermphanaphan

import java.util.Scanner;

class Node {
    private int val;
    private Node left;
    private Node right;

    public Node(int data){
        this.val = data;
        this.left = null;
        this.right = null;
    }

    public Node getLeft(){
        return this.left;
    }

    public Node getRight(){
        return this.right;
    }

    public void setLeft(Node n){
        this.left = n;
    }

    public void setRight(Node n){
        this.right = n;
    }

    public int getVal(){
        return this.val;
    }
}

class Tree {
    private Node root;

    public Tree(){
        this.root = null;
    }

    public void insertNode(int val){
        Node n = new Node(val);

        if (this.root == null){
            this.root = n;
            return;
        }

        Node cur = this.root;
        while (true){
            if (cur.getVal() == n.getVal()) return;
            if (cur.getVal() < n.getVal()){
                if (cur.getRight() == null){
                    cur.setRight(n);
                    return;
                }
                cur = cur.getRight();
            }
            else {
                if (cur.getLeft() == null){
                    cur.setLeft(n);
                    return;
                }
                cur = cur.getLeft();
            }

        }
    }

    private void inorder(Node n){
        if (n != null){
            inorder(n.getLeft());
            System.out.printf("%d ", n.getVal());
            inorder(n.getRight());
        }
    }

    public void res(){
        inorder(this.root);
        System.out.printf("\n");
    }
}

public class Lab06_BST {
    public static void main(String[] args) {
        Tree x = new Tree();
        String usrInp;
        Scanner sc = new Scanner(System.in);
        while (true){
            usrInp = sc.next();
            if (usrInp.equals("END")){
                break;
            }

            int y = Integer.parseInt(usrInp);
            x.insertNode(y);
        }
        sc.close();
        x.res();
    }
}