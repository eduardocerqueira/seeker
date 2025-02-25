//date: 2025-02-25T16:52:37Z
//url: https://api.github.com/gists/573697b980e252ba6a96afb3c0ce6163
//owner: https://api.github.com/users/itssamsa

public class Estudiante {

    private String nombre;
    private int edad;
    private String genero;
    private String documento;
    private String alergias;
    private String acudiente;
    private String contacto;

    public Estudiante(String nombre, int edad, String genero, String documento, String alergias, String acudiente, String contacto) {
        this.nombre = nombre;
        this.edad = edad; 
        this.genero = genero;
        this.documento = documento;
        this.alergias = alergias;
        this.acudiente = acudiente;
        this.contacto = contacto;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public int getEdad() {
        return edad;
    }

    public void setEdad(int edad) {
            this.edad = edad;
    }

    public String getGenero() {
        return genero;
    }

    public void setGenero(String genero) {
        this.genero = genero;
    }

    public String getDocumento() {
        return documento;
    }

    public void setDocumento(String documento) {
        this.documento = documento;
    }

    public String getAlergias() {
        return alergias;
    }

    public void setAlergias(String alergias) {
        this.alergias = alergias;
    }

    public String getAcudiente() {
        return acudiente;
    }

    public void setAcudiente(String acudiente) {
        this.acudiente = acudiente;
    }

    public String getContacto() {
        return contacto;
    }

    public void setContacto(String contacto) {
        this.contacto = contacto;
    }

    @Override
    public String toString() {
        return "Estudiante{" +
                "nombre='" + nombre + '\'' +
                ", edad=" + edad +
                ", genero='" + genero + '\'' +
                ", documento='" + documento + '\'' +
                ", alergias='" + alergias + '\'' +
                ", acudiente='" + acudiente + '\'' +
                ", contacto='" + contacto + '\'' +
                '}';
    }
}