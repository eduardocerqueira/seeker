//date: 2022-09-30T17:08:15Z
//url: https://api.github.com/gists/c8317c218be19d6366f7daec39100b20
//owner: https://api.github.com/users/yazmin-erazo

package EntradaBufferedInputStream;

import java.io.*;

public class EntradaDatosBufferedInputStream {
  public static void main(String[] args) {
    // FileInputStream = devuelve una secuencia de bytes por eso se debe guardar en el tipo de datos acorde
    try{
      InputStream fichero = new FileInputStream("C:\\Users\\kevin\\Desktop\\fichero.txt");
      // Con BufferedStream se crea una copia por pedacitos y el nos va pasando la informaci[on
      //Cuando se crea BufferedInputStream, se crea una matriz de búfer interna.
      BufferedInputStream ficheroBuffer = new BufferedInputStream(fichero);

      try{
        int dato = ficheroBuffer.read();
        while(dato != -1){
          System.out.print((char)dato);
          dato = ficheroBuffer.read();
        }





      }catch (IOException e){
        System.out.println("No puedo leer el fichero: " + e.getMessage());
      }

    }catch(FileNotFoundException e){
      System.out.println("Oye, que el programa da error: " + e.getMessage());
    }

  }
}
