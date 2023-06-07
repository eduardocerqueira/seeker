//date: 2023-06-07T16:40:23Z
//url: https://api.github.com/gists/3b939c91e7c0ec628dec6ad1c55a983b
//owner: https://api.github.com/users/sebastianbarrionuebo

import Clases.Caja;
import Clases.Cliente;
import Clases.FilaEspera;
import Enums.FormaDePago;
import Enums.Tipo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.awt.*;
import java.io.*;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        FilaEspera<Cliente> filaPrincipal = new FilaEspera<Cliente>();
        Random dados = new Random();

        Cliente yo = new Cliente(Tipo.COMUN, FormaDePago.EFECTIVO,dados.nextInt(10,40));
        Cliente tu = new Cliente(Tipo.JUBILADO,FormaDePago.TARJETASPROBLEMAS,dados.nextInt(10,40));
        Cliente el = new Cliente(Tipo.EMBARASADA,FormaDePago.TARJETACPROBLEMAS,dados.nextInt(10,40));
        Cliente dio = new Cliente(Tipo.COMUN,FormaDePago.TARJETASPROBLEMAS,dados.nextInt(10,40));
        Cliente jojo = new Cliente(Tipo.JUBILADO,FormaDePago.EFECTIVO,dados.nextInt(10,40));
        Cliente stan = new Cliente(Tipo.EMBARASADA,FormaDePago.TARJETACPROBLEMAS,dados.nextInt(10,40));

        Caja caja1 = new Caja();
        Caja caja2 = new Caja();

        caja1.entraCliente(yo);
        caja1.entraCliente(el);
        caja2.entraCliente(tu);
        caja2.entraCliente(dio);

        filaPrincipal.llegaCliente(jojo);
        filaPrincipal.llegaCliente(stan);

        System.out.println("Caja1 " + caja1);
        System.out.println("Caja2 " + caja2);

        try{
            System.out.println("Tiempo en caja1: " + caja1.atender());
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        try{
            System.out.println("Tiempo en caja2: " + caja2.atender());
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        System.out.println("Caja1 " + caja1);
        System.out.println("Caja2 " + caja2);


        Boolean ope = true;
        while (!filaPrincipal.filaVacia()){
            if(ope){
                caja1.entraCliente(filaPrincipal.getSiguiente());
                ope = false;
            }else{
                caja2.entraCliente(filaPrincipal.getSiguiente());
                ope = true;
            }
        }
        System.out.println("Caja1 " + caja1);
        System.out.println("Caja2 " + caja2);

        ///Json
        try{
            pasarAJson(caja1.devolverEnListaClientes());
        }catch (IOException e){
            System.out.println(e.getMessage());
        }
        try{
            System.out.println( descargarDeJson() );
        }catch (IOException e){
            System.out.println(e.getMessage());
        }

        System.out.println("Hello world!");
    }

    ///JackSon
    public static void pasarAJson(List<Cliente> lista) throws IOException {
        File archivo = new File("D:\\Java\\Java Projets\\Progra3-3P\\src\\Archivos\\lista.json");
        ObjectMapper buffer = new ObjectMapper();

        buffer.writerWithDefaultPrettyPrinter().writeValue(archivo,lista);
    }

    public static List descargarDeJson() throws IOException {
        File archivo = new File("D:\\Java\\Java Projets\\Progra3-3P\\src\\Archivos\\lista.json");
        ObjectMapper buffer = new ObjectMapper();

        List<Cliente> lista = buffer.readValue(archivo,List.class);
        return lista;
    }


    ///Gson
    /*
    public static List<Autor> descargarJson() throws IOException {
        File archivo = new File("src\\Archivos\\generated.json");
        BufferedReader buffer = new BufferedReader(new FileReader(archivo));
        Gson gson = new Gson();
        Type listaAutor = "**********"
        return gson.fromJson(buffer, listaAutor);
    }

    public static void pasarAJson(Caja caja) throws IOException {
        File archivo = new File("D:\\Java\\Java Projets\\Progra3-3P\\src\\Archivos\\lista.json");
        BufferedWriter buffer = new BufferedWriter(new FileWriter(archivo));
        Gson gson = new Gson();
        String jsonString = "";

        jsonString = gson.toJson(caja);
        try{
            buffer.write(jsonString);
            buffer.flush();
            buffer.close();
        }catch (IOException e){
            System.out.println(e.getMessage());
        }finally {
            buffer.close();
        }
    }
    */

}close();
        }
    }
    */

}