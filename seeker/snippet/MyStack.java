//date: 2023-01-12T17:13:21Z
//url: https://api.github.com/gists/ac5def684fb38a3e291090cea540426b
//owner: https://api.github.com/users/OguzhanEfendi

public class MyStack {
    Integer value;


    private int size=0;
    MyStack next;

    public MyStack(int value) {
        this.value = value;
        size++;
    }

    public MyStack(int value, MyStack next) {
        this.value = value;
        this.next = next;
        size++;
    }
    public void   push(int newValue){
       MyStack newStack = new MyStack(this.value,this.next);
       this.value=newValue;
       this.next=newStack;
       size++;
    }

    public Integer pop(){

        if (size>0) {
            size--;
            int poppedValue = this.value;
            if (this.next != null) {
                this.value = this.next.value;
                this.next = this.next.next;
                return poppedValue;
            } else if (this.value!=null) {
                this.value=null;
                return poppedValue;
            }
        }
        else
            System.out.println("Stack is empty!");
        return null;
    }

    public boolean isEmpty(){
        return size==0;
    }

    @Override
    public String toString() {
        String stackValues="";
        MyStack myStack = this;
        while (myStack!=null){
            stackValues +=myStack.value+" ";
            myStack=myStack.next;
        }
        return stackValues;
    }

    public int getSize() {
        return size;
    }

}
