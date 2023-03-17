//date: 2023-03-17T17:07:52Z
//url: https://api.github.com/gists/48a147277e2baa68b76a74d1f29ed91b
//owner: https://api.github.com/users/nunchy-pc


package PROG06_PradanosAnunciacion;

/**
 * clase " Reparacion " con los atributos descripcion y matrícula.
 *
 * @author nunchy
 */
public class Reparacion {

    private String descripcion;
    private String matricula;

// Constructor sin parámetros
    public Reparacion() {
    }
    
    
 // Constructor con el parámetro descripcion
    public Reparacion(String descripcion) {

        this.descripcion = descripcion;
    }
    //métodos get y set  con los  parámetros descripcion y matricula
    public String getDescripcion() {
        return descripcion;
    }

    public void setDescripcion(String descripcion) {
        this.descripcion = descripcion;
    }

    public String getMatricula() {
        return matricula;
    }

    public void setMatricula(String matricula) {
        this.matricula = matricula;
    }
    //método toString que devuelve la descripcion
    @Override
    public String toString() {
        return descripcion;
    }

}
