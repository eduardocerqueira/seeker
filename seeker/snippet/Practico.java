//date: 2025-04-10T17:06:21Z
//url: https://api.github.com/gists/9f4d909b9f5a33986a755fe7070f55d6
//owner: https://api.github.com/users/kurotori

import java.util.Scanner;

public class Practico {
    public static void main(String[] args){
        Scanner teclado = new Scanner(System.in);
        String nombreU="";
        String opcion = "";
        int cantidadDatos = 10;
        String[] nombres = new String[cantidadDatos];
        String[] apellidos = new String[cantidadDatos];
        String[] emails = new String[cantidadDatos];
        boolean[] enUso = new boolean[cantidadDatos];

        for (int i = 0; i < cantidadDatos; i++) {
            nombres[i] = "";
            apellidos[i] = "";
            emails[i] = "";
            enUso[i] = false;
        }
        
        System.out.println("Hola. Por favor ingresa tu nombre.");
        System.out.print("Nombre:\t");
        nombreU = teclado.nextLine();

        while ( ! opcion.equals("X")) {
            System.out.println("Hola, "+nombreU);
            System.out.println("Por favor, elije una opción:");
            System.out.println("A - Agregar Datos al Sistema");
            System.out.println("V - Ver Todos los Datos Almacenados");
            System.out.println("B - Buscar y Mostrar un Dato");
            System.out.println("E - Eliminar Todos los Datos");
            System.out.println("X - Salir");
            System.out.print("Opción:\t");
            opcion = teclado.nextLine();
            opcion = opcion.toUpperCase();
            
            switch (opcion) {
                
                case "X":
                    System.out.println(nombreU + ", vas a cerrar el programa");
                    System.out.println("¿Desea continuar?");
                    System.out.println("S - Si");
                    System.out.println("N - No");
                    System.out.print("Opción:\t");
                    opcion = teclado.nextLine();
                    opcion = opcion.toUpperCase();
                    if (opcion.equals("S")) {
                        opcion = "X";
                        System.out.println("Adios, " + nombreU);    
                    }
                    break;
                
                case "A":
                    int posicionLibre = cantidadDatos;
                    for (int i = 0; i < cantidadDatos; i++) {
                        if ( enUso[i] == false ) {
                            posicionLibre = i;
                            break;
                        }
                    }
                    if (posicionLibre < cantidadDatos) {
                        
                        System.out.println("Ingresa un registro nuevo a los datos");
                        System.out.print("Nombre:\t\t");
                        nombres[posicionLibre] = teclado.nextLine();
                        System.out.print("Apellido:\t");
                        apellidos[posicionLibre] = teclado.nextLine();
                        System.out.print("E-Mail:\t\t");
                        emails[posicionLibre] = teclado.nextLine();
                        enUso[posicionLibre] = true;
                        System.out.println("Se ha agregado el registro en la posicion " + posicionLibre);
                    }
                    else{
                        System.out.println("No quedan posiciones libres en el programa");
                        System.out.println("Presione Enter para continuar");
                        opcion=teclado.nextLine();
                        opcion="";
                    }
                    break;
                
                case "V":
                    int registros = 0;
                    
                    for (int i = 0; i < cantidadDatos; i++) {
                        if ( enUso[i] == true ) {
                            registros++;
                        }
                    }

                    if (registros > 0) {
                        System.out.println("Datos Almacenados:");
                        System.out.println("N°\tNombre\t\tApellido\t\tE-Mail");
                        for (int i = 0; i < cantidadDatos; i++) {
                            if (enUso[i] == true) {
                                System.out.printf(
                                "%d)\t%1.15s\t%1.15s\t%1.15s%n",
                                i,nombres[i],apellidos[i],emails[i]
                                );    
                            }
                                
                        }
                        
                    }
                    else{
                        System.out.println("No hay datos almacenados en el programa");
                    }
                    System.out.println("Presione Enter para continuar");
                    opcion=teclado.nextLine();
                    opcion="";
                    break;
                
                case "B":
                    System.out.println(nombreU + " ¿Qué dato buscas?");
                    System.out.print("Dato a buscar:\t");
                    String datoBusqueda = teclado.nextLine();
                    System.out.println("Resultados:");
                    for (int i = 0; i < cantidadDatos; i++) {
                        if (enUso[i] == true) {
                            if (nombres[i].contains(datoBusqueda)) {
                                System.out.printf(
                                "%d)\t%1.15s\t%1.15s\t%1.15s%n",
                                i,nombres[i],apellidos[i],emails[i]
                                );    
                            }
                            if (apellidos[i].contains(datoBusqueda)) {
                                System.out.printf(
                                "%d)\t%1.15s\t%1.15s\t%1.15s%n",
                                i,nombres[i],apellidos[i],emails[i]
                                );    
                            }
                            if (emails[i].contains(datoBusqueda)) {
                                System.out.printf(
                                "%d)\t%1.15s\t%1.15s\t%1.15s%n",
                                i,nombres[i],apellidos[i],emails[i]
                                );    
                            }
                            else{
                                System.out.println("No se han encontrado resultados");
                            }
                        }
                    }
                    System.out.println("Presione Enter para continuar");
                    opcion=teclado.nextLine();
                    opcion="";
                    break;
                
                case "E":
                    System.out.println( nombreU + " se van a eliminar TODOS los registros" );
                    System.out.println("¿Desea continuar?");
                    System.out.println("S - Si");
                    System.out.println("N - No");
                    System.out.print("Opción:\t");
                    opcion = teclado.nextLine();
                    opcion = opcion.toUpperCase();
                    if (opcion.equals("S")) {
                        for (int i = 0; i < cantidadDatos; i++) {
                            nombres[i] = "";
                            apellidos[i] = "";
                            emails[i] = "";
                            enUso[i] = false;
                        }
                        System.out.println("Se borraron todos los registros");
                        System.out.println("Presione Enter para continuar");
                        opcion=teclado.nextLine();
                        opcion="";   
                    }
                    break;
                
                default:
                    System.out.println("ERROR: No es una opción válida");
                    System.out.println("Presione Enter para continuar");
                    opcion=teclado.nextLine();
                    opcion="";
                    break;
            }
        }
        
        teclado.close();
    }
}
