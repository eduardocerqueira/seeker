//date: 2022-09-21T17:22:18Z
//url: https://api.github.com/gists/af5f5c6442b222be1ffa5ccddf45f156
//owner: https://api.github.com/users/miqueiasousa

import java.util.Comparator;

public class SinglyLinkedList<E> {
  private Node<E> head;
  private Node<E> tail;
  private int size = 0;

  public void add(E e) {
    Node<E> node = new Node<E>(e);

    if (isEmpty()) {
      head = node;
    }

    if (size == 1) {
      head.next = node;
    }

    if (size > 1) {
      tail.next = node;
    }

    tail = node;
    size += 1;
  }

  public void clear() {
    head = null;
    tail = null;
    size = 0;
  }

  public boolean contains(E e) {
    if (isEmpty()) {
      return false;
    }

    Node<E> current = head;
    boolean contains = false;

    while (current != null) {
      if (current.element.equals(e)) {
        contains = true;
        break;
      }

      current = current.next;
    }

    return contains;
  }

  public E get(int i) {
    E element = null;
    int headIndex = 0;
    int tailIndex = size() - 1;

    if (i >= size()) {
      System.out.println("Index is out of range");

      return null;
    }

    if (i == headIndex) {
      return head.element;
    }

    if (i == tailIndex) {
      return tail.element;
    }

    Node<E> current = head.next;

    for (int j = 1; j < tailIndex; j++) {
      element = current.element;

      if (j == i) {
        break;
      } else {
        current = current.next;
      }
    }

    return element;
  }

  public boolean isEmpty() {
    return size == 0;
  }

  public void pop() {
    if (size() <= 1) {
      clear();

      return;
    }

    Node<E> current = head;

    while (current.next.next != null) {
      current = current.next;
    }

    current.next = null;
    tail = current;
    size -= 1;
  }

  public void remove(int i) {
    int headIndex = 0;
    int tailIndex = size() - 1;

    if (i >= size()) {
      System.out.println("Index is out of range");

      return;
    }

    if (i == headIndex) {
      shift();

      return;
    }

    if (i == tailIndex) {
      pop();

      return;
    }

    Node<E> current = head;

    for (int j = 1; j < tailIndex; j++) {
      if (j == i) {
        current.next = current.next.next;
      } else {
        current = current.next;
      }
    }

    size -= 1;
  }

  public void shift() {
    if (size() <= 1) {
      clear();

      return;
    }

    head = head.next;
    size -= 1;
  }

  public int size() {
    return size;
  }

  public void sort(Comparator<E> cond) {
    if (size() <= 1) {
      return;
    }

    Node<E> current = head;

    while (current != null) {
      Node<E> index = current.next;

      while (index != null) {
        if (cond.compare(current.element, index.element) > 0) {
          E temp = current.element;
          current.element = index.element;
          index.element = temp;
        }

        index = index.next;
      }

      current = current.next;
    }
  }

  public String toString() {
    StringBuilder str = new StringBuilder();

    if (isEmpty()) {
      str.append("[]");

      return str.toString();
    }

    str.append("[");

    Node<E> current = head;

    while (current != null) {
      str.append(current.element);

      if (current.next != null) {
        str.append(", ");
      }

      current = current.next;
    }

    str.append("]");

    return str.toString();
  }
}
