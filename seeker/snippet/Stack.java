//date: 2021-12-01T17:07:43Z
//url: https://api.github.com/gists/bb293c1d83c4004bbed06112b029aaf0
//owner: https://api.github.com/users/cbkpar

public
class Stack<E> extends Vector<E> {
  public Stack() {
  }
  
  public E push(E item) {
      addElement(item);
      return item;
  }
  
  public synchronized E pop() {
    E       obj;
    int     len = size();
    obj = peek();
    removeElementAt(len - 1);
    return obj;
  }
  
  public synchronized E peek() {
    int     len = size();
    if (len == 0)
        throw new EmptyStackException();
    return elementAt(len - 1);
  }
  
  public boolean empty() {
    return size() == 0;
 }
  
  public synchronized int search(Object o) {
    int i = lastIndexOf(o);
    if (i >= 0) {
        return size() - i;
    }
    return -1;
 }
}