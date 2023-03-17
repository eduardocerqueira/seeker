//date: 2023-03-17T17:07:52Z
//url: https://api.github.com/gists/48a147277e2baa68b76a74d1f29ed91b
//owner: https://api.github.com/users/nunchy-pc

package PROG06_PradanosAnunciacion;

import java.util.Scanner;
import static PROG06_PradanosAnunciacion.Taller.validarMatricula;

/**
 * La clase Principal muestra el menú y leería la entrada del usuario hasta que
 * se seleccione la opción "Salir" (6) El switch determinaría la opción
 * seleccionada y llama al método correspondiente.
 *
 * @author nunchy
 */
public class Principal {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        Scanner teclado = new Scanner(System.in);

        boolean salir = false;/* variable booleana utilizada en el bucle while que controla
        la ejecución del programa, permitiendo que el usuario pueda salir del mismo desde el menu principal.*/

        String opcion; //  variable String que se utiliza para recoger la opción seleccionada por el usuario en el menú principal del programa.
        Taller taller = new Taller();/*instancia de la clase Taller que se utiliza para almacenar 
        y gestionar los vehículos y las reparaciones.*/
        Vehiculo vehiculo = null;/*variable de tipo Vehiculo que se utiliza para almacenar el vehículo 
        que se va a buscar, modificar o eliminar del taller.*/
        Reparacion reparacion = null;/* variable de tipo Reparacion que se utiliza para almacenar 
        la nueva reparación que se va a agregar a un vehículo existente en el taller.*/


 /* el bucle while se ejecutará siempre y cuando la variable "salir" sea falsa.
        Una vez que la variable cambie a verdadera, el bucle se detendrá */
        while (!salir) {

            /*se está utilizando un bloque try-catch para manejar excepciones.
            En particular, se está tratando de asegurar que la opción de menu ingresada por el usuario
            sea un valor numérico entre 1 y 7.  */
            try {
                //Se llama al método mostrarMenu;
                Taller.mostrarMenu();
                // se recoge el dato quitando espacios en blanco al principio y al final;
                opcion = teclado.nextLine().trim();
                //verificamos que cumple los requisitos con el método validarOpcion, si no lo cumple se lanza excepcion
                if (!Taller.validarOpcion(opcion)) {
                    throw new Exception("No ha introducido un valor numerico valido entre el 1 y el 7.");
                }
                /* switch evalúa una única expresión y ejecuta el bloque de código correspondiente al valor de la expresión. */
                switch (opcion) {
                    case "1":
                        /* Se comprueba si la cantidad de vehículos, si es mayor o igual a 5  se imprime mensaje */
                        if (taller.getCantidadVehiculos() >= 5) {
                            System.out.println("------------------------------------------------------------");
                            System.out.println("El taller esta completo. No se pueden agregar mas vehiculos.");
                            System.out.println("------------------------------------------------------------");
                            // si es inferior imprime mensaje solicitando datos
                        } else {
                            System.out.println("--------------------------------------");
                            System.out.println("Ingrese los datos del nuevo vehiculo:");
                            System.out.println("--------------------------------------");
                            // variables String que se utilizan para recoger los datos de un nuevo vehículo que se van a agregar al taller
                            String marca = "";
                            String modelo = "";
                            String color = "";
                            String matricula = "";
                            /*Para cada recogida de datos coloco un while para que no quedeel dato vacio 
                            pero no se tendrá que ingresar de nuevo los datos que sí se han ha completado*/
                            while (marca.isEmpty()) {
                                System.out.println("Marca: (Campo obligatorio)");
                                marca = teclado.nextLine();
                            }

                            while (modelo.isEmpty()) {
                                System.out.println("Modelo: (Campo obligatorio)");
                                modelo = teclado.nextLine();
                            }
                            while (color.isEmpty()) {
                                System.out.println("Color: (Campo obligatorio)");
                                color = teclado.nextLine();
                            }
                            /* en este último bucle recogemos el dato pasandolo a mayusculas y sin espacios al principio ni al final,
                               e introducimos el condicionante de que si no valida la matricula con el método validadMatricula() imprima 
                                 mensaje indicando el formato correcto y la matrícula vuelve a estar vacia para retomar el bucle*/
                            while (matricula.isEmpty()) {
                                System.out.println("Matricula: (Campo obligatorio)");
                                matricula = teclado.nextLine().toUpperCase().trim();
                                if (!validarMatricula(matricula)) {
                                    taller.mensajeMatriculaIncorrecta(matricula);
                                    matricula = "";
                                    /*superada las restricciones se llama al método agregarVehiculo() del objeto "taller" 
                                    con los datos ingresados para agregar el nuevo vehículo a la lista de vehículos del taller*/
                                } else {
                                    taller.agregarVehiculo(marca, modelo, color, matricula);
                                }
                            }

                        }
                        break;

                    case "2":
                        // se llama al método listarVehiculos() para mostrar el listado de vehículos:
                        taller.listarVehiculos();
                        break;

                    case "3":
                        // se solicita al usuario que ingrese una matrícula de vehículo
                        System.out.println("------------------------------------");
                        System.out.println("Ingrese la matricula del vehiculo: ");
                        System.out.println("------------------------------------");
                        //se recoge el dato pasandolo a mayúsculas y sin espacios en blanco al pricipio y al final
                        String matriculaBuscada = teclado.nextLine().toUpperCase().trim();
                        // llamada al método para buscar un vehículo por matrícula
                        taller.buscarVehiculo(matriculaBuscada);
                        // Si la matrícula ingresada no es válida según la función "validarMatricula", se muestra un mensaje de error.
                        if (!validarMatricula(matriculaBuscada)) {
                            taller.mensajeMatriculaIncorrecta(matriculaBuscada);
                            //Si no se encuentra un vehículo con esa matrícula, se muestra un mensaje con la información del vehículo. 
                        } else if (taller.buscarVehiculo(matriculaBuscada) == null) {
                            System.out.println("No se encontro ningun vehiculo con la matrícula " + matriculaBuscada);
                            //Si se encuentra un vehículo con esa matrícula, se muestra un mensaje con la información del vehículo.
                        } else {
                            System.out.println("-----------------------");
                            System.out.println("Vehiculo encontrado:");
                            System.out.println(taller.buscarVehiculo(matriculaBuscada));

                        }

                        break;
                    case "4":
                        // llamar al método para agregar una nueva reparación a un vehículo
                        System.out.println("------------------------------------");
                        System.out.println("Ingrese la matricula del vehiculo:");
                        System.out.println("------------------------------------");
                        //se recoge el dato pasandolo a mayúsculas y sin espacios en blanco al pricipio y al final
                        String matriculaReparacion = teclado.nextLine().toUpperCase().trim();

                        // se llama al método buscarVehiculo();
                        taller.buscarVehiculo(matriculaReparacion);
                        // se valida la matrícula,si no es válida se imprime mensaje de error
                        if (!validarMatricula(matriculaReparacion)) {
                            taller.mensajeMatriculaIncorrecta(matriculaReparacion);
                            // se comprueba si el vehículo tiene menos de tres reparaciones registradas
                            //con el método taller.comprobarCantidadReparaciones
                        } else if (taller.comprobarCantidadReparaciones(matriculaReparacion)) {
                            //si el  vehículo tiene menos de tres reparaciones , se solicita una descripción de la nueva reparación
                            String descripcion = "";
                            while (descripcion.isEmpty()) {
                                System.out.println("Ingrese una descripcion de la reparacion:   (Campo obligatorio)");
                                descripcion = teclado.nextLine();
                            }

                            //  System.out.println("Ingrese una descripcion de la reparacion:");
                            //  String descripcion = teclado.nextLine();
                            //if (descripcion == null || descripcion.trim().isEmpty()) {
                            //    System.out.println("No puede ingresar una descripcion vacia");
                            taller.agregarReparacion(matriculaReparacion, descripcion);

                        } else {
                            System.out.println("Vehiculo " + matriculaReparacion + " con el cupo de 3 reparaciones . No se pueden agregar mas reparaciones.");
                        }

                        break;

                    case "5":
                        //Se solicita ingresa la matricula;
                        System.out.println("-----------------------");
                        System.out.println("Ingrese la matricula del vehiculo: ");
                        //se recoge el dato pasandolo a mayúsculas y sin espacios en blanco al pricipio y al final

                        String matriculaListarReparaciones = teclado.nextLine().toUpperCase().trim();
                        // se valida la matrícula,si no es válida se imprime mensaje de error

                        if (!validarMatricula(matriculaListarReparaciones)) {
                            taller.mensajeMatriculaIncorrecta(matriculaListarReparaciones);

                            //  si valida la matricula llama al método para mostrar el listado de reparaciones de un vehículo
                        } else {

                            taller.listarReparaciones(matriculaListarReparaciones);
                        }
                        break;

                    case "6":
                        System.out.println("------------------------------------");
                        System.out.println("Ingrese la matricula del vehiculo: ");
                        System.out.println("------------------------------------");
                        //se recoge el dato pasandolo a mayúsculas y sin espacios en blanco al pricipio y al final
                        String matriculaEliminarVehiculo = teclado.nextLine().toUpperCase().trim();
                        // se llama al método eliminarVehiculo;
                        // se valida la matrícula,si no es válida se imprime mensaje de error

                        if (!validarMatricula(matriculaEliminarVehiculo)) {
                            taller.mensajeMatriculaIncorrecta(matriculaEliminarVehiculo);

                            //  si valida la matricula llama al método para mostrar el listado de reparaciones de un vehículo
                        } else {

                            taller.eliminarVehiculo(matriculaEliminarVehiculo);
                        }

                        break;

                    case "7":
                        salir = true;// la variable toma el valor para salir del programa
                        System.out.println("Saliendo del programa");
                        break;
                    default:// en caso de introducir una opcion que no esté recogido en el switch
                        System.out.println("Opcion invalida");
                        break;
                }
                // se cierra el bloque try-catch
            } catch (Exception e) {
                System.out.println(e.getMessage());

            }
        }
    }
}
