//date: 2023-03-17T17:07:52Z
//url: https://api.github.com/gists/48a147277e2baa68b76a74d1f29ed91b
//owner: https://api.github.com/users/nunchy-pc

package PROG06_PradanosAnunciacion;

/**
 * Clase "Taller" contiene un array de objetos "Vehiculo" con capacidad máxima
 * de 5 y un array de objetos "Reparaciones" con máximo de 3. Tiene un método get para la cantidad de vehiculo.
 * También tiene métodos para mostrar el menu principal, validar la opcion de menu ,
 * validar la matrícula, mostrar mensaje si la matricula no es válida, agregar
 * un nuevo vehículo al taller,colocarlo en la primera posicion libre del array,
 * buscar un vehículo por matrícula, indicar si el vehículo ya está en el taller,
 * mostrar el listado de vehículos,agregar una nueva reparación a un vehículo, ,
 * mostrar el listado de reparaciones de un vehículo y eliminar un vehículo
 *
 * @author nunchy
 */
public class Taller {

   //array de tipo Vehiculo que tiene una longitud de 5 elementos. Será utilizado para almacenar los objetos  "Vehiculo" que se irán agregando al taller.
    private Vehiculo[] vehiculos = new Vehiculo[5];
    // variable entera que inicialmente está en 0 y se utilizará para llevar un registro de la cantidad de objetos  "Vehiculo" que hay en el array "vehiculos".
    private int cantidadVehiculos = 0;
    //array de tipo "Reparacion" con una longitud de 3 elementos.Será utilizado para almacenar las reparaciones que se realicen en los vehículos del taller
    private Reparacion[] reparaciones = new Reparacion[3];
    //variable entera que inicialmente está en 0 y se utilizará para llevar un registro de la cantidad de objetos de tipo "Reparacion" que hay en el array "reparaciones".
    private int cantidadReparaciones = 0;

    // Constructor vacio:
    public Taller() {
    }
    // método get para obtener la cantidad de vehículos
    public int getCantidadVehiculos() {
        return cantidadVehiculos;
    }

    /**
     * método mostrarMenu imprime las diferentes opciones del menú inicial
     */
    public static void mostrarMenu() {
        System.out.println("--------------------------------");
        System.out.println("Menu Taller mecanico Nunchy:");
        System.out.println("1. Nuevo vehiculo");
        System.out.println("2. Listado de vehiculos");
        System.out.println("3. Buscar vehiculo");
        System.out.println("4. Nueva reparacion");
        System.out.println("5. Listado de reparaciones");
        System.out.println("6. Eliminar vehiculo");
        System.out.println("7. Salir \n");
        System.out.print("Selecciona una opcion: \n");
    }

    /**
     * método validarOpcion verifica si una cadena de texto (en este caso, la
     * variable opcion) cumple con las condiciones para considerarla una opción
     * válida en el menú del taller. La condición es que la cadena no sea nula
     * (opcion != null) y que contenga solo caracteres numéricos del 1 al 7
     * (opcion.matches("[1-7]+")).
     *
     * @param opcion cadena que ha de cumplir los requisitos
     * @return devuelve true si es válida. Si no cumple alguna de las dos
     * condiciones, el método devuelve false, indicando que la opción no es
     * válida.
     */
    public static boolean validarOpcion(String opcion) {
        if (opcion != null && opcion.matches("[1-7]+")) {
            return true;
        }
        return false;

    }

 

    /**
     * método "validarMatricula" recibe una cadena matricula que representa una
     * posible matrícula de vehículo, y verifica si se ajusta a un patrón
     * específico. El patrón verifica si la matrícula tiene 4 dígitos seguidos
     * de 3 letras mayúsculas que no incluyen vocales.
     *
     * @param matricula variable cadena que ha de cumplir el patrón.
     * @return true si la matrícula cumple con el patrón y false si no lo hace.
     */
    public static boolean validarMatricula(String matricula) {

        if (matricula.matches("^[0-9]{4}[BCDFGHJKLMNPQRSTVWXYZ]{3}$")) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * método mensajeMatriculaIncorrecta imprime un mensaje en la consola
     * indicando que la matrícula ingresada no es válida.
     *
     * @param matricula cadena que no cumple los requisitos Imprime matrícula
     * ingresada y explicación de cómo debe ser el formato de la matrícula
     * válida.
     */
    public void mensajeMatriculaIncorrecta(String matricula) {
        System.out.println("La matricula ingresada " + matricula + " no es valida.\n Debe tener el formato NNNNLLL,"
                + " donde N es un digito entre 0 y 9 y L es una letra consonante mayuscula");

    }

    /**
     * método " comprobarCantidadVehiculo" que agrega un nuevo vehículo al array
     * de vehículos si la capacidad del taller no está completa suma uno mas a
     * la cantidad de vehículos del taller. Si el taller está completo lo indica
     * a través de un mensaje
     *
     * @param vehiculo objeto de la clase Vehiculo.
     */
    public void comprobarCantidadVehiculo(Vehiculo vehiculo) {
        if (cantidadVehiculos < 5) {
            vehiculos[cantidadVehiculos] = vehiculo;
            cantidadVehiculos++;
        } else {
            System.out.println("El taller esta completo. No se pueden agregar más vehiculos.");
        }
    }

    /**
     * recorre el array de vehículos hasta encontrar una posición libre. Si
     * encuentra una posición libre, entonces busca el siguiente vehículo no
     * es nulo y lo mueve a la posición libre que encontró antes. Luego, sigue
     * buscando posiciones libres hasta encontrar la primera. Si no hay
     * posiciones libres, retorna -1.
     *
     * @param vehiculos array de Vehiculo
     * @return i para la posicion libre y -1 si no hay posiciones libres
     *
     */
    public int buscarPosicionLibre(Vehiculo[] vehiculos) {
        for (int i = 0; i < vehiculos.length; i++) {
            if (vehiculos[i] == null) {
                // Recolocamos los vehículos
                for (int j = i + 1; j < vehiculos.length; j++) {
                    if (vehiculos[j] != null) {
                        vehiculos[i] = vehiculos[j];
                        vehiculos[j] = null;
                        break;
                    }
                }
                return i;
            }
        }
        return -1;
    }

 
    /**
     * método agregarVehiculo verifica si el vehículo ya se encuenra en el
     * taller , agrega un nuevo vehículo al taller si la cantidad de vehículos
     * es menor a 5 y lo agrega en la primera posicion libre. Si la cantidad de
     * vehículos es igual a 5, no se puede agregar más vehículos al taller y se
     * imprime un mensaje indicando que el taller está completo.
     *
     * @param marca
     * @param modelo
     * @param color
     * @param matricula. Cadenas requeridas para agregar un nuevo vehículo.
     */
    public void agregarVehiculo(String marca, String modelo, String color, String matricula) {

        if (vehiculoEnTaller(matricula)) {
            System.out.println("-------------------------------------------------------------------------");
            System.out.println("El vehiculo con matricula " + matricula + " ya se encuentra en el taller. \n");
            //una vez superadas las restricciones agregamos el vehiculo al taller:
        } else if ((cantidadVehiculos < 5)) {
            Vehiculo nuevoVehiculo = new Vehiculo(marca.toUpperCase().trim(), modelo.toUpperCase().trim(), color.toUpperCase().trim(), matricula.toUpperCase().trim());
            comprobarCantidadVehiculo(nuevoVehiculo);
            int posicionLibre = buscarPosicionLibre(vehiculos);
            if (posicionLibre != -1) {
                vehiculos[posicionLibre] = nuevoVehiculo;

                System.out.println("------------------------------------------");
                System.out.println("Vehiculo agregado al taller exitosamente.");
                System.out.println("------------------------------------------");
            } else {
                System.out.println("------------------------------------------------------------");
                System.out.println("Vehiculo agregado al taller exitosamente.");
                System.out.println("Actualmente hay " + cantidadVehiculos);
                System.out.println("------------------------------------------------------------");
            }
        } else {
            System.out.println("------------------------------------------------------------");
            System.out.println("El taller esta completo. No se pueden agregar más vehiculos.");
            System.out.println("------------------------------------------------------------");
        }
    }

    /**
     * método "buscarVehiculo" busca un vehículo en el array de vehículos por su
     * matrícula y si la localiza devuelve el objeto "Vehiculo" correspondiente.
     *
     * @param matricula cadena requerida para poder localizar un vehiculo.
     * @return vehiculos con la matricula indicada.
     */
    public Vehiculo buscarVehiculo(String matricula) {
        for (int i = 0; i < cantidadVehiculos; i++) {
            if (vehiculos[i].getMatricula().equals(matricula)) {
                return vehiculos[i];
            }
        }
        return null;
    }

    /**
     * Este método recorre el array de vehículos buscando la matrícula indicada
     * para verificar si el vehiculo se encuentra en el taller
     *
     * @param matricula cadena requerida para poder localizar un vehiculo.
     * @return true, indicando que el vehículo se encuentra en el taller. De lo
     * contrario, devuelve false
     */
    public boolean vehiculoEnTaller(String matricula) {
        for (int i = 0; i < cantidadVehiculos; i++) {
            if (vehiculos[i].getMatricula().equals(matricula)) {
                return true;
            }
        }
        return false;
    }

    /**
     * método "listarVehiculos" que recorre el array de vehículos y muestra los
     * datos de cada uno y sus reparaciones. Si la cantidad de vechiculos es
     * mayor que 0 imprime mensaje con los datos de los vehículos, si no es
     * mayor que 0 imprime mensaje indicando que no hay vehiculos en el taller.
     */
    public void listarVehiculos() {
        if (cantidadVehiculos > 0) {
            System.out.println("-----------------------");
            System.out.println("Listado de vehiculos:");
            System.out.println("-----------------------");
            for (int i = 0; i < cantidadVehiculos; i++) {
                System.out.println("Vehiculo " + (i + 1) + ": " + vehiculos[i].toString());
            }
        } else {
            System.out.println("-------------------------------");
            System.out.println("No hay vehiculos en el taller.");
           
        }
    }
/**
 *  Método comprobarCantidadReparaciones() comprueba si un vehículo tiene menos de 3 reparaciones registradas.
     * @param matricula cadena requerida para poder localizar un vehiculo.
     * @return devuelve devuelve false si la cantidad de reparaciones es>=3. 
     *         devuelve truesi la cantidad es inferior a 3
 */

    public boolean comprobarCantidadReparaciones(String matricula) {
        Vehiculo vehiculo = buscarVehiculo(matricula);
        if (vehiculo.getCantidadReparaciones() >= 3) {
            //System.out.println("El vehiculo con matricula " + matricula + " ya tiene el maximo de reparaciones.");
            return false;
        }
        return true;
    }


    /**
     * método agregarReparacion se encarga de agregar una nueva reparación a un vehículo en la lista de reparaciones
     * de dicho vehículo.
     * Primero crea una nueva instancia de la clase Reparación con la descripción  y  la agrega a la lista de reparaciones
     * del vehículo correspondiente utilizando el método nuevaReparacion() de la clase Vehículo, que se encarga de agregar
     * la nueva reparación a la lista.
     * Finalmente, imprime un mensaje indicando que la reparación ha sido agregada .
     *
     * @param matricula cadena requerida para poder localizar un vehiculo.
     * @param descripcion para crear una nueva reparacion y agregarla al vehiculo con la matricula indicada
     */

    public void agregarReparacion(String matricula, String descripcion) {
     
        Reparacion nuevaReparacion = new Reparacion(descripcion);
        buscarVehiculo(matricula).nuevaReparacion(nuevaReparacion);
        System.out.println("Reparacion agregada al vehiculo con matricula " + matricula + ".");
    }




    /**
     * * método listarReparaciones recibe una matrícula de un vehículo y busca
     * en el taller si existe un vehículo con esa matrícula.Si se encuentra un
     * vehículo, entonces se imprime la información del vehículo y se recorreel
     * array de reparaciones para imprimir la descripción de cada reparación que
     * tenga el vehículo. Si no se encuentra un vehículo con la matrícula
     * ingresada, se imprime un mensaje indicando que no se encontró ningún
     * vehículo con esa matrícula.
     *
     * @param matricula cadena requerida para poder localizar un vehiculo.
     *
     */
    public void listarReparaciones(String matricula) {
        Vehiculo vehiculo = buscarVehiculo(matricula.toUpperCase());
        if (vehiculo != null) {
            System.out.println("Reparaciones del vehiculo con matricula " + matricula + ":");
            Reparacion[] reparaciones = vehiculo.getReparaciones();
            System.out.println(vehiculo);
            for (int i = 0; i < reparaciones.length; i++) {
                if (reparaciones[i] != null) {
                    System.out.println("Reparacion " + (i + 1) + ": " + reparaciones[i].getDescripcion());
                } else {
                    System.out.println("Reparacion " + (i + 1) + ":   Vehiculo sin reparacion numero " + (i + 1) + " agregada.");
                }
            }
        } else {
            System.out.println("No se encontro ningun vehiculo con la matricula " + matricula);
        }
    }

    /**
     * método eliminarReparaciones busca el vehículo a través de su matrícula y,
     * si lo encuentra, recorre el array de reparaciones asociadas a ese
     * vehículo y las elimina y resetea la cantidad de reparaciones del vehículo
     * a cero. Si no encuentra ningún vehículo con la matrícula especificada,
     * muestra un mensaje indicándolo.
     *
     * @param matricula cadena requerida para poder localizar un vehiculo.
     *
     */
    public void eliminarReparaciones(String matricula) {
        Vehiculo vehiculo = buscarVehiculo(matricula);
        if (vehiculo != null) {
            Reparacion[] reparaciones = vehiculo.getReparaciones();
            for (int i = 0; i < reparaciones.length; i++) {
                if (reparaciones[i] != null) {
                    reparaciones[i] = null; // Eliminamos la reparación
                }
            }
            vehiculo.setCantidadReparaciones(0); // Reseteamos la cantidad de reparaciones del vehículo
            System.out.println("Reparaciones eliminadas del vehiculo con matricula " + matricula);
        } else {
            System.out.println("No se encontro ningún vehículo con la matricula " + matricula);
        }
    }

    /**
     * método eliminarVehiculo comprueba si el vehículo a eliminar está en el taller, elimina sus
     * reparaciones, luego busca su posición en el array de vehículos para
     * eliminarlo y recolocar los demás vehículos en el array.
     *
     * @param matriculaEliminar cadena requerida para poder localizar un
     * vehiculo.
     * @return
     */
    public Vehiculo eliminarVehiculo(String matriculaEliminar) {
        if (!vehiculoEnTaller(matriculaEliminar.toUpperCase())) {
            System.out.println("El vehiculo con matricula " + matriculaEliminar + " no se encuentra en el taller. ");
        } else {
            eliminarReparaciones(matriculaEliminar.toUpperCase());
            int posicionAEliminar = -1;
            for (int i = 0; i < vehiculos.length; i++) {
                if (vehiculos[i] != null && vehiculos[i].getMatricula().equals(matriculaEliminar.toUpperCase())) {
                    posicionAEliminar = i;
                }
            }
            if (posicionAEliminar >= 0) {
                vehiculos[posicionAEliminar] = null;
                for (int i = posicionAEliminar; i < vehiculos.length - 1; i++) {
                    vehiculos[i] = vehiculos[i + 1];
                }
                vehiculos[vehiculos.length - 1] = null;
                cantidadVehiculos--;
                System.out.println("El vehículo con matricula " + matriculaEliminar.toUpperCase() + " ha sido eliminado del taller.");

            } else {
                System.out.println("Error al eliminar el vehículo con matricula " + matriculaEliminar + ".");

            }
        }

        return null;

    }
    
     

}
