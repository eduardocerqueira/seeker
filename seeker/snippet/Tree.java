//date: 2021-12-06T16:48:35Z
//url: https://api.github.com/gists/8cbd563539960427c993090d589cff2c
//owner: https://api.github.com/users/begimkulmarzhan

package tree;

import exception.NodeException;

public class Tree implements TreeOperations {
  private Tree right; // класс объектісі параметр
  private Tree left; //  класс объектісі параметр
  private int value; //  айнымалы параметр

  /**
   * empty constructor
   * sets node's value to 0
   */
  public Tree() {
    this.value = 0;
  } // конструктор

  /**
   * this constructor sets this node's value to given parameter
   * @param value setting value
   */
  public Tree(int value) {
    this.value = value;
  } // Конструктор

  /**
   * this method adds a new given node to this tree
   * @param node given new node
   */
  @Override
  public void addNode(Tree node){  // Ағашқа жаңа нүктені қосу әдісі
    if(node.getValue() < this.value){ // шартты оператор
      if(this.left == null) { // шартты оператор Егер де ағаштың сол жақ бөлігі бос болса
        this.left = node;
      } else {
        this.getLeft().addNode(node); // ағаштың сол жағына элемент қосу
      }

    } else {
      if(this.right == null) {  // Оң жағы бос болса
        this.right = node;
      } else {
        this.getRight().addNode(node); // оң жағына қосу элементті
      }
    }
  }


  /**
   * this method adds a new node with given value to this tree
   * @param val given new node's value
   */
  @Override
  public void addNode(int val){ // Ағаш бөлігіне айнымалыны қосу

    if(val < this.value){ // Ағаштының оң жағын бөлігін және сол жақ бөлігін салыстыру
      if(this.left == null) { // Ағаштың сол жақ бөлігіне элемент қосу
        this.left = new Tree(val);
      } else {
        this.getLeft().addNode(val); // Сол жақ бөлігіне қосу
      }

    } else {
      if(this.right == null) { // Ағаштың оң жағына бөлігіне қосу әдісі
        this.right = new Tree(val);
      } else {
        this.getRight().addNode(val);
      }
    }
  }

  /**
   * this method removes the node that contains given value from this tree
   * @param val given value
   * @throws NodeException if given value doesn't exist in this tree
   */
  @Override
  public void removeNode(int val) throws NodeException { // Ағаш бөлігінен элементті өшіру әдісі
    Tree node = findNodeForRemove(val);
    node.setRight(null);
    node.setLeft(null);

  }

  /**
   * this method is called in 'removeNode' method
   * it finds a node that has children node containing given value
   * @param val removing node's value
   * @return returns node that has children node containing given value
   * if that node doesn't exist this method will return abstract node
   * @throws NodeException if given value doesn't exist in this tree
   */
  @Override
  public Tree findNodeForRemove(int val) throws NodeException { // Ағаш бөлігінен элементті өшіру үшін алдымен ағаш нүктесін анықтау қажет
    if(!nodeExists(this, val)){ // Шартты оператор
      throw new NodeException("this value doesn't exist in this tree!!!"); // Қателік
    }
    Tree node = this;
    Tree r = node.getRight(); // Оң жақ бөлігі бойынша санды алу
    Tree l = node.getLeft(); // Сол жақ бөлігі бойынша санды алу

    if((l != null && val == l.getValue()) || (r != null && val == r.getValue())){
      return node;
    }

    if(l != null && val < l.getValue()) // Шартты оператор сол жақ бөлігі бос немесе енгізілген сннан үлкен болса
      return l.findNodeForRemove(val); // Сол санды қайтура керек

    if(r != null && val >= r.getValue()) // Ағаш бөлігінің оң жақ бөлігі бос және енгізілген санға тен немесе үлкен болса
      return r.findNodeForRemove(val);

    return node;
  }

  /**
   * this method finds a node containing given value
   * @param val given value
   * @return the node containing given value
   * if that node doesn't exist this method will cause stackOverFlow
   * @throws NodeException if given value doesn't exist in this tree
   */
  @Override
  public Tree findNode(int val) throws NodeException { // Ағаш бөлігінен нүктені іздеу әдісі
    if(!nodeExists(this, val)){
      throw new NodeException("this value doesn't exists in this tree!!!"); // Қателік түрі
    }
    Tree node = this; // Объект класса Tree

      if(val == node.getValue()) // Шартты оператор енгізілген сан нүктедегі мәнге тең болса
        return this; // Санды қайтар

      if(val < node.getValue()){ // Егер сан кіші болса
        if(this.getLeft() != null) // Және нүкте бос болмаса
          node = this.getLeft(); // сол жақ бөлігіне тең
      } else {
        if(this.getRight() != null) // Егер он жақ бөлігі бос болмаса
          node = this.getRight(); // он жақ бөлігіне тең
      }

    return node.findNode(val); // Табылған санды қайтару
  }

  /**
   * checks recursively if there exists a given value in this tree
   * @param node searching tree
   * @param val searching value
   * @return "true" if there exists a given value in this tree, returns "false" otherwise
   */
  public boolean nodeExists(Tree node, int val){ // Нүктенің бар жоғын анықтайтын әдіс
    if (node == null) // егер нүкті бос болса
      return false; // логикалық false қайтару

    if (node.getValue() == val) // Егер нүкте қандайда бір санға тең болса
      return true; // True қайтар

    /* then recur on left subtree */
    boolean res1 = nodeExists(node.getLeft(), val);
    // node found, no need to look further
    if(res1) return true;

    /* node is not found in left,
    so recur on right subtree */
    boolean res2 = nodeExists(node.getRight(), val);

    return res2;
  }

  public Tree getRight() {
    return right;
  }


  public void setRight(Tree right) {
    this.right = right;
  }

  public Tree getLeft() {
    return left;
  }

  public void setLeft(Tree left) {
    this.left = left;
  }

  public int getValue() {
    return value;
  }

  public void setValue(int value) {
    this.value = value;
  }



  @Override
  public String toString() {
    return "Tree{" +
        "value=" + value +
        ", left=" + left +
        ", right="+ right + '}';
  }
}
