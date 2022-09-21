//date: 2022-09-21T17:22:18Z
//url: https://api.github.com/gists/af5f5c6442b222be1ffa5ccddf45f156
//owner: https://api.github.com/users/miqueiasousa

public class Node<E> {
  public E element;
  public Node<E> next = null;

  public Node(E e) {
    element = e;
  }
}
