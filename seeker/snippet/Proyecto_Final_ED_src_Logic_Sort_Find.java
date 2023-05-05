//date: 2023-05-05T17:08:46Z
//url: https://api.github.com/gists/9592131b57c831576a6ef5c856697f3e
//owner: https://api.github.com/users/frankxhunter

package Logic;

import cu.edu.cujae.ceis.tree.binary.BinaryTree;
import cu.edu.cujae.ceis.tree.binary.BinaryTreeNode;
import cu.edu.cujae.ceis.tree.iterators.binary.PreorderIterator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.ListIterator;


public class Sort_Find{
	
	public static HashMap<String, Letra> contarFrecuenciaCaracteres(String input) {
		String[] caracteres = input.split("");

		HashMap<String, Letra> frecuencias = new HashMap<>();

		for (String caracter : caracteres)
			if (frecuencias.containsKey(caracter)) 
				frecuencias.get(caracter).aumentaFrecuencia();
			 else 
				frecuencias.put(caracter, new Letra(caracter,1));
			
		
		return frecuencias;
	}
	
	public static void mergeSort(ArrayList<Letra> letras, int left, int right) {
        if (left < right) {
            int middle = (left + right) / 2;
            mergeSort(letras, left, middle);
            mergeSort(letras, middle + 1, right);
            merge(letras, left, middle, right);
        }
    }

    private static void merge(ArrayList<Letra> letras, int left, int middle, int right) {
        ArrayList<Letra> temp = new ArrayList<>(letras.subList(left, right + 1));

        int i = 0;
        int j = middle - left + 1;
        int k = left;

        while (i <= middle - left && j <= right - left) {
            if (temp.get(i).compareTo(temp.get(j)) <= 0) {
                letras.set(k, temp.get(i));
                i++;
            } else {
                letras.set(k, temp.get(j));
                j++;
            }
            k++;
        }

        while (i <= middle - left) {
            letras.set(k, temp.get(i));
            k++;
            i++;
        }
    }


    //Metodos para crear arbol
	public static BinaryTree<String> crearArbol(ArrayList<Letra> letras, Hashtable<String,String> abecedario) {
		BinaryTreeNode<String> root = new BinaryTreeNode<>(null);
		BinaryTree<String> tree = new BinaryTree<>(root);
		ArrayList<Conjunto> conjuntos = Convertir_a_Conjunto(letras);
		UnirConjuntos(conjuntos);
		BinaryTreeNode<String> left = new BinaryTreeNode<>(null);
		BinaryTreeNode<String> right = new BinaryTreeNode<>(null);
		root.setLeft(left);
		root.setRight(right);
		crearArbol(left, abecedario,"1", conjuntos.get(0));
		crearArbol(right, abecedario,"0", conjuntos.get(1));
		return tree;
	}

	private static void crearArbol(BinaryTreeNode<String> node,Hashtable<String,String> abecedario,String codigo, Conjunto conj) {
		if (conj.letra == null) {
			BinaryTreeNode<String> left = new BinaryTreeNode<>(null);
			BinaryTreeNode<String> right = new BinaryTreeNode<>(null);
			node.setLeft(left);
			node.setRight(right);
			crearArbol(left, abecedario,codigo+"1", conj.conj1);
			crearArbol(right, abecedario,codigo+"0", conj.conj2);
		} else {
			String letra=conj.letra.getLetra();
			node.setInfo(letra);
			abecedario.put(letra,codigo);
		}
	}

	private static ArrayList<Conjunto> Convertir_a_Conjunto(ArrayList<Letra> letras) {
		ArrayList<Conjunto> salida = new ArrayList<Conjunto>();
		for (Letra x : letras)
			salida.add(new Conjunto(x));

		return salida;
	}

	private static void UnirConjuntos(ArrayList<Conjunto> conj) {
		while (conj.size() > 2) {
			Conjunto c = new Conjunto(conj.remove(0), conj.remove(0));
			ListIterator<Conjunto> it = conj.listIterator(conj.size());
			boolean encontrado = false;
			while (it.hasPrevious() && !encontrado)
				if (it.previous().compareTo(c) != 1)
					encontrado = true;

			it.next();
			it.add(c);
		}
	}
//Metodo para codificar
	public static String encodeWord(String word,Hashtable<String,String> abecedario){
		String salida="";
		for(int i=0; i<word.length();i++)
			salida=salida+abecedario.get(word.substring(i,i+1));

		return salida;
}
//Metodo para decodificar
public static String decode(BinaryTree<String> tree,String codigo) {
	String result = "";
	int i = 0, tamano = codigo.length();
	BinaryTreeNode<String> aux;
	BinaryTreeNode<String> aux_root = (BinaryTreeNode<String>) tree.getRoot();
	while (i < tamano) {
		aux = aux_root;
		if (codigo.charAt(i) == '1') {
			if (tree.nodeIsLeaf(aux.getLeft())) {
				result = result + aux.getLeft().getInfo();
				aux_root = (BinaryTreeNode<String>) tree.getRoot();
			} else {
				aux_root = aux.getLeft();
			}
		} else {
			if (tree.nodeIsLeaf(aux.getRight())) {
				result = result + aux.getRight().getInfo();
				aux_root = (BinaryTreeNode<String>) tree.getRoot();
			} else {
				aux_root = aux.getRight();
			}
		}
		i++;
	}
	if (aux_root == null)
		result = result + ('-');
	return result;
}






	public static void main(String[] args) {
	String prueba="Este es una cadena utilizada para saber la frecuencia de aparicion de cada caracter 1234";
//		String prueba= "Holaaaallo";
	HashMap<String, Letra> mapa= Sort_Find.contarFrecuenciaCaracteres(prueba);
	ArrayList<Letra> listaLetras=new ArrayList<>(mapa.values());
	
	System.out.println(prueba);
	System.out.println("Frecuencia sin ordenar:");
	for(Letra l: listaLetras)
		System.out.print(l.getLetra()+": "+l.getFrecuencia()+"  ");
	System.out.println();
	
	Sort_Find.mergeSort(listaLetras, 0, listaLetras.size()-1);
	
	System.out.println("Frecuencia ordenda y lista para construir el arbol de Huffman:");
	for(Letra l: listaLetras)
		System.out.print(l.getLetra()+": "+l.getFrecuencia()+"  ");

	System.out.println("Arbol de Huffman recorrido en preorden");
	Hashtable<String, String> abecedario= new Hashtable<String, String>();
	BinaryTree treeHuffman= crearArbol(listaLetras,abecedario);
	PreorderIterator<String> it=treeHuffman.preOrderIterator();
	while(it.hasNext()) {
		BinaryTreeNode<String> x=it.nextNode();
		System.out.print(x.getInfo()==null? "-1":"Letra:"+x.getInfo());
		System.out.print("  //");
	}
	System.out.println("\n");
//	String mensaje="Hola"
		String mensaje = prueba;
	String codigo=encodeWord(mensaje, abecedario);
	System.out.println("Codificando la frase: "+mensaje);
	System.out.println(codigo);


	//Decodificando
		mensaje=decode(treeHuffman, codigo);
		System.out.println("Decodificando dicho codigo: "+ mensaje);

	System.out.println("Fin");

}
}

class Letra implements Comparable<Letra> {
	private String letra;
	private int frecuencia;

	public Letra(String letra, int frecuencia) {
		this.setLetra(letra);
		this.setFrecuencia(frecuencia);
	}
	public int getFrecuencia() {
		return frecuencia;
	}

	public void setFrecuencia(int frecuencia) {
		this.frecuencia = frecuencia;
	}

	public String getLetra() {
		return letra;
	}

	public void setLetra(String letra) {
		this.letra = letra;
	}
	public void aumentaFrecuencia() {
		frecuencia++;
	}

	@Override
	public int compareTo(Letra o) {
		return Integer.compare(this.frecuencia, o.getFrecuencia());
	}
}



class Conjunto implements Comparable<Conjunto> {
	Conjunto conj1;
	Conjunto conj2;
	Letra letra;
	int suma;

	public Conjunto(Conjunto conj1, Conjunto conj2) {
		this.conj1 = conj1;
		this.conj2 = conj2;
		letra = null;
		suma = conj1.suma + conj2.suma;
	}

	public Conjunto(Letra l) {
		this.letra = l;
		suma = letra.getFrecuencia();
		conj1 = null;
		conj2 = null;
	}

	public int compareTo(Conjunto otraConjunto) {
		return Integer.compare(this.suma, otraConjunto.suma);
	}

}

