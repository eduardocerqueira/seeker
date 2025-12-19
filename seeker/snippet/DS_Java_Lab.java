//date: 2025-12-19T17:01:06Z
//url: https://api.github.com/gists/75ef4e4922b03e0915d83546f962cd35
//owner: https://api.github.com/users/bizzarejojo392-a11y

1. Linear Search
Linear search sequentially checks each element until the required key is found. Time Complexity:
O(n).
class LinearSearch {
    public static void main(String[] args) {
        int[] a = {10, 20, 30, 40};
        int key = 30;
        for (int i = 0; i < a.length; i++) {
            if (a[i] == key) {
                System.out.println("Found at index " + i);
                return;
            }
        }
        System.out.println("Not found");
    }
}
Binary Search
Binary search works on sorted arrays and divides the search space into two halves. Time
Complexity: O(log n).
class BinarySearch {
    public static void main(String[] args) {
        int[] a = {10, 20, 30, 40, 50};
        int key = 40, l = 0, r = a.length - 1;
        while (l <= r) {
            int m = (l + r) / 2;
            if (a[m] == key) {
                System.out.println("Found at index " + m);
                return;
            } else if (a[m] < key)
                l = m + 1;
            else
                r = m - 1;
        }
        System.out.println("Not found");
    }
}
2. Bubble Sort
Bubble sort repeatedly swaps adjacent elements if they are in wrong order. Time Complexity: O(n²).
class BubbleSort {
    public static void main(String[] args) {
        int[] a = {5, 3, 4, 1};
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a.length - i - 1; j++)
                if (a[j] > a[j + 1]) {
                    int t = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = t;
                }
        for (int x : a)
            System.out.print(x + " ");
    }
}
Selection Sort
Selection sort selects the minimum element and places it at correct position. Time Complexity:
O(n²).
class SelectionSort {
    public static void main(String[] args) {
        int[] a = {64, 25, 12};
        for (int i = 0; i < a.length; i++) {
            int min = i;
            for (int j = i + 1; j < a.length; j++)
                if (a[j] < a[min])
                    min = j;
            int t = a[min];
            a[min] = a[i];
            a[i] = t;
        }
        for (int x : a)
            System.out.print(x + " ");
    }
}
Insertion Sort
Insertion sort inserts elements into their proper position in sorted part. Time Complexity: O(n²).
class InsertionSort {
    public static void main(String[] args) {
        int[] a = {9, 5, 1};
        for (int i = 1; i < a.length; i++) {
            int key = a[i], j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
        for (int x : a)
            System.out.print(x + " ");
    }
}
Quick Sort
Quick sort uses divide and conquer strategy with pivot element. Average Time Complexity: O(n log
n).
class QuickSort {
    static void quick(int[] a, int l, int h) {
        if (l < h) {
            int p = part(a, l, h);
            quick(a, l, p - 1);
            quick(a, p + 1, h);
        }
    }
    static int part(int[] a, int l, int h) {
        int p = a[h], i = l - 1;
        for (int j = l; j < h; j++)
            if (a[j] < p) {
                i++;
                int t = a[i]; a[i] = a[j]; a[j] = t;
            }
        int t = a[i + 1]; a[i + 1] = a[h]; a[h] = t;
        return i + 1;
    }
    public static void main(String[] args) {
        int[] a = {10, 7, 8};
        quick(a, 0, a.length - 1);
        for (int x : a)
            System.out.print(x + " ");
    }
}
3. Singly Linked List
Linked list is a dynamic data structure consisting of nodes connected using links.
class LinkedList {
    static class Node {
        int data;
        Node next;
        Node(int d) { data = d; }
    }
    public static void main(String[] args) {
        Node head = new Node(10);
        head.next = new Node(20);
        head.next.next = new Node(30);
        Node t = head;
        while (t != null) {
            System.out.print(t.data + " ");
            t = t.next;
        }
    }
}
4. Stack using Array
Stack follows Last In First Out (LIFO) principle.
class StackArray {
    int top = -1;
    int[] s = new int[5];
    void push(int x) { s[++top] = x; }
    int pop() { return s[top--]; }
    public static void main(String[] args) {
        StackArray st = new StackArray();
        st.push(10);
        st.push(20);
        System.out.println(st.pop());
    }
}
5. Queue using Array
Queue follows First In First Out (FIFO) principle.
class QueueArray {
    int[] q = new int[5];
    int f = 0, r = 0;
    void enqueue(int x) { q[r++] = x; }
    int dequeue() { return q[f++]; }
    public static void main(String[] args) {
        QueueArray q = new QueueArray();
        q.enqueue(10);
        q.enqueue(20);
        System.out.println(q.dequeue());
    }
}
6. Binary Tree
Binary tree allows maximum two children per node. Operations include traversal, height and sum.
class BinaryTree {
    static class Node {
        int data;
        Node left, right;
        Node(int d) { data = d; }
    }
    static int sum(Node r) {
        if (r == null) return 0;
        return r.data + sum(r.left) + sum(r.right);
    }
    public static void main(String[] args) {
        Node root = new Node(10);
        root.left = new Node(5);
        root.right = new Node(15);
        System.out.println("Sum = " + sum(root));
    }
}