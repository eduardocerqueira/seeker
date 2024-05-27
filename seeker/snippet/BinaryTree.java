//date: 2024-05-27T16:46:50Z
//url: https://api.github.com/gists/8917ef912ddfcc37ba6045c7850d9496
//owner: https://api.github.com/users/jairoArh

package logic;

import logic.Node;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BinaryTree <T>{

    private Comparator<T> comparator;
    private Node<T> root;

    private List<T> list;

    public BinaryTree(Comparator<T> comparator) {
        root = null;
        this.comparator = comparator;
    }

    public boolean isEmpty(){

        return root == null;
    }

    public void addNode(T info){
        if( isEmpty( ) ){
            root = new Node<>(info);
        }else{
            Node<T> father = null;
            Node<T> aux = root;
            while( aux != null ){
                father = aux;
                aux = comparator.compare( aux.getInfo(),info) > 0 ? aux.getLeft() : aux.getRight();
            }
            Node<T> nodeNew = new Node<>( info );
            if( comparator.compare( father.getInfo(),info) > 0 ){
                father.setLeft( nodeNew );
            }else{
                father.setRight( nodeNew );
            }

        }
    }

    public List<T> listPresort( ){
        list = new ArrayList<>();

        presort( root );

        return list;
    }

    private void presort(Node<T> root) {
        if( root != null ){
            list.add( root.getInfo( ) );
            presort( root.getLeft( ) );
            presort( root.getRight( ) );
        }
    }

    public List<T> listInsort(){
        list = new ArrayList<>();

        insort( root );

        return list;
    }

    private void insort(Node<T> root) {
        if( root != null ){
            insort( root.getLeft( ) );
            list.add( root.getInfo( ) );
            insort( root.getRight( ) );
        }
    }

    public List<T> listAmplitudeDown(){
        list = new ArrayList<>();
        //TODO cola para aregar los elementos que se van a procesar


        return list;
    }

    /**
     * Método que busca un elemento en el árbol binario,
     * @param info elemento a buscar (indicar el valor de la clave)
     * @return La referencia del Nodo que contiene el elemento
     */
    public Node<T> findNode( T info){

        Node<T> aux = root;
        while( aux != null ){
            if( comparator.compare(info,aux.getInfo()) == 0 ){

                return aux;
            }
            aux = comparator.compare( info, aux.getInfo()) > 0 ? aux.getRight() : aux.getLeft();
        }

        return null;
    }

    public Node<T> findFather( Node<T> node ){
        if( node == root ){
            return null;
        }else{
            Node<T> aux = root;
            while( true ){
                if( aux.getRight() == node || aux.getLeft() == node ){

                    return aux;
                }
                aux = comparator.compare( node.getInfo(), aux.getInfo()) > 0 ? aux.getRight() : aux.getLeft();
            }
        }
    }
}
