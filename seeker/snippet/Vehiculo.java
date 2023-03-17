//date: 2023-03-17T17:07:52Z
//url: https://api.github.com/gists/48a147277e2baa68b76a74d1f29ed91b
//owner: https://api.github.com/users/nunchy-pc


package PROG06_PradanosAnunciacion;
/**
 * clase "Vehiculo" con los atributos marca, modelo, color y matrícula.
 * La clase "Vehiculo" también tendrá un array de objetos "Reparacion" con capacidad máxima de 3 para almacenar
 * sus reparaciones.
 * @author nunchy
 */
public class Vehiculo {
    
    
    private String marca;
    private String modelo;
    private String color;
    private String matricula;
    /*Matriz de objetos de la clase Reparacion, con un tamaño máximo de 3,
    que representa las reparaciones que se han hecho al vehículo.  */
    private Reparacion[] reparaciones = new Reparacion[3];
    // un entero que indica la cantidad de reparaciones que se han hecho al vehículo.
    private int cantidadReparaciones = 0;
    
        // Constructor sin parámetros
    public Vehiculo() {
    }
    // Constructor con los  parámetros marca, modelo, color y  matrícula del vehículo.
    public Vehiculo(String marca, String modelo, String color, String matricula) {
        this.marca = marca;
        this.modelo = modelo;
        this.color = color;
        this.matricula = matricula;
     
    }
    
    // métodos get y set correspondientes para cada propiedad.

    public String getMarca() {
        return marca;
    }

    public void setMarca(String marca) {
        this.marca = marca;
    }

    public String getModelo() {
        return modelo;
    }

    public void setModelo(String modelo) {
        this.modelo = modelo;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public String getMatricula() {
        return matricula;
    }

    public void setMatricula(String matricula) {
        this.matricula = matricula;
    }

  
    public Reparacion[] getReparaciones() {
        return reparaciones;
    }

    public void setReparaciones(Reparacion[] reparaciones) {
        this.reparaciones = reparaciones;
    }
    public int getCantidadReparaciones() {
        return cantidadReparaciones;
    }

    public void setCantidadReparaciones(int cantidadReparaciones) {
        this.cantidadReparaciones = cantidadReparaciones;
    }
    
    
    /**
     * El método nuevaReparacion agrega una nueva reparación al vehículo, siempre y cuando no se hayan hecho
     * ya 3 reparaciones.Si el vehículo ya tiene 3 reparaciones, se mostrará un mensaje indicando que no se pueden 
     * agregar más reparaciones.
     * @param reparacion
     */
    
    public void nuevaReparacion(Reparacion reparacion) {
        if (cantidadReparaciones < 3) {
            reparaciones[cantidadReparaciones] = reparacion;
            cantidadReparaciones++;
        } else {
            System.out.println("No se pueden agregar mas reparaciones a este vehículo.");
        }
       

  
    
    
       
    }
/**
 *  método toString retorna una cadena que representa el vehículo, incluyendo  marca, modelo, color y matrícula, 
 * 
*/
    @Override
    public String toString() {
        
         
   
        return "\n MARCA: " + marca + "\n MODELO: " + modelo + " \n COLOR: " + color + "\n MATRICULA: " + matricula.toUpperCase();
    }
  
    
    
    
        
}
