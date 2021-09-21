//date: 2021-09-21T17:01:41Z
//url: https://api.github.com/gists/8bf0c2617d8d0b59f365f145c6058b32
//owner: https://api.github.com/users/BoruE23

public class LinkList<E>
{
   Node head;
   Node last;
   int size;
   
     public LinkList()
    {
      head = null;
      last = null;
      size = 0;
    }
    
    public void add(E data){
        size++;
        Node nNode = new Node(data);
        
        if(head == null){
            head = nNode;
            last = nNode;
        }else{
            last.setNext(nNode);
            last = nNode;
        }
        }
    
    public void add(int index, E data){
      size++;
      Node nNode = new Node(data);
      Node cn = head;
      Node prev = head;
      
      if(head == null){
          head = nNode;
          last = nNode;
      }else{
          for(int i=0; i<index;i++){
            prev = cn;
            cn = cn.next;
        }
        prev.next = nNode;
        nNode.next = cn;
      }
      
      
    }
    
    public E get(int index){
        if(index == 0){
            return (E) head.data;
        }
        
        Node currNode = head;
        
        for(int i=0;i<index;i++){
            if(currNode.next !=null){
                currNode = currNode.next;
            }
            }
            return (E) currNode.data;
        }
        
        
    public int size(){
        return size;
    }
    
    public E remove(int index){
        Node cn = head;
        Node prev = head;
        if(index > size || index < 0){
         System.out.println("Out of Bounds");
         return null;
        }
        size--;
        
        if(index == 0){
            cn.next = head;
        }else{
        for(int i=0; i<index;i++){
            prev = cn;
            cn = cn.next;
        }
        
        prev.next = prev.next.next;
        }
        
        return (E) cn.data;
    }
    
    public void set(int index, E e){
    Node cn = head;
    for(int i = 0; i< index; i++){
        cn = cn.next;
    }
    cn.data = e;
   }
   
   public void addToFront(E e){
       size++;
       Node nnode = new Node(e);
       nnode.next = head;
    
   }
   
   public E removeFront(){
       size--;
       Node nnode;
       nnode = head;
       Node cn = head;
       head = nnode.next;
       return (E) cn.data;
    
   }
   
   public int contains(E e){
       Node cn = head;
       for(int i=0;i<size;i++){
        if(cn.data.equals(e)){
            return i;
        }
        cn=cn.next;
       }
       return -1;
       }
   
    
   }
    

    
 
        
       

        
    