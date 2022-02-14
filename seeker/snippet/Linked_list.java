//date: 2022-02-14T16:55:01Z
//url: https://api.github.com/gists/459e87f7e3980623b9f0c704ff2164aa
//owner: https://api.github.com/users/krish233

import java.util.Scanner;

public class Linked_list {
    class Node{
        String data;int data1;
        Node next;

        public Node(String data) {
            this.data = data;

            this.next = null;
        }
        public Node(int data1){
            this.data1= data1;
            this.next=null;
        }
    }

    //Represent the head and tail of the singly linked list
    public Node head = null;
    public Node tail = null;

    //addNode() will add a new node to the list
    public void addNode(String data) {
        //Create a new node
        Node newNode = new Node(data);

        //Checks if the list is empty
        if(head == null) {
            //If list is empty, both head and tail will point to new node
            head = newNode;
            tail = newNode;
        }
        else {
            //newNode will be added after tail such that tail's next will point to newNode
            tail.next = newNode;
            //newNode will become new tail of the list
            tail = newNode;
        }
    }
    public void addNodeInt(int data) {
        //Create a new node
        Node newNode = new Node(data);

        //Checks if the list is empty
        if(head == null) {
            //If list is empty, both head and tail will point to new node
            head = newNode;
            tail = newNode;
        }
        else {
            //newNode will be added after tail such that tail's next will point to newNode
            tail.next = newNode;
            //newNode will become new tail of the list
            tail = newNode;
        }
    }

    //display() will display all the nodes present in the list
    public void display() {
        //Node current will point to head
        Node current = head;

        if(head == null) {
            System.out.println("List is empty");
            return;
        }
        System.out.println("Nodes of singly linked list: ");
        while(current != null) {
            //Prints each node by incrementing pointer
            System.out.print(current.data + " ");
            current = current.next;
        }
        while (current.next!=null){
            System.out.println(current.data1+" ");
        }
        System.out.println();
    }
    public static void main(String[] args) {
        Scanner s=new Scanner(System.in);
        Linked_list ls=new Linked_list();
        String str[]= new String[s.nextInt()];
        for (int i = 0; i < str.length; i++) {
            str[i]=s.next();
        }
        String temp[]=new String[str.length];
        try{
        for (int i = 0; i < str.length; i++) {

                temp[i]="";
                if(Integer.parseInt(str[i])>0)
                {
                    for (int j = 0; j < str[i].length(); j++)
                    {
                        int l1=Integer.parseInt(String.valueOf(str[i].charAt(j)));


                        if(l1==0){
                            if (Integer.parseInt(String.valueOf(str[i].charAt(0)))==0)
                                continue;
                            temp[i]=temp[i]+"z";
                            continue;
                        }
                        if(l1%2==0)
                            temp[i]=temp[i]+"e";
                        if(l1%2!=0)
                            temp[i]=temp[i]+"o";
                    }
                    ls.addNode("P"+temp[i]);
                }

                if(Integer.parseInt(str[i])<0)
                {for (int k = 1; k < str[i].length(); k++)
                {
                    int l=Integer.parseInt(String.valueOf(str[i].charAt(k)));
                    if(l==0){
                        if (Integer.parseInt(String.valueOf(str[i].charAt(1)))==0)
                            continue;
                        temp[i]=temp[i]+"z";
                        continue;
                    }
                    if(l%2==0)
                        temp[i]=temp[i]+"e";
                    if(l%2!=0)
                        temp[i]=temp[i]+"o";
                }
                    ls.addNode("N"+temp[i]);
                }
        }
    } catch (Exception o)
        {
            o.printStackTrace();
        }
        ls.display();
    }
}
