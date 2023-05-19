//date: 2023-05-19T16:41:12Z
//url: https://api.github.com/gists/eecda6a30850642a3ef6bd5302cc0f31
//owner: https://api.github.com/users/kevinber789

package Transporte;
import java.util.Calendar;
import java.util.Date;

public class Fecha {
  int a�o, mes, dia, hora, minuto;
  Fecha() {
		 Calendar calendario=Calendar.getInstance();  //Calendar: clase abstracta
		 Date fecha=calendario.getTime();
		 calendario.setTime(fecha);
		 a�o=calendario.get(Calendar.YEAR);
		 mes=calendario.get(Calendar.MONTH)+1;
		 dia=calendario.get(Calendar.DAY_OF_MONTH);
		 hora=calendario.get(Calendar.HOUR_OF_DAY);
		 minuto=calendario.get(Calendar.MINUTE);
	}

  	boolean menor90minutos(Fecha fPosterior) {
  		boolean ok=false;
  		int auxh=hora*60+minuto+90;
  		int auxhPost= fPosterior.hora*60+fPosterior.minuto;
  		System.out.println(fPosterior.hora+":"+fPosterior.minuto+",,"
  		  +auxh+" ->"+auxhPost);
  		if (a�o==fPosterior.a�o && mes==fPosterior.mes && 
  				dia==fPosterior.dia && auxhPost<=auxh)
  			ok=true;
  		return ok;	
  	}
	public Fecha(int a�o, int mes, int dia, int hora, int minuto) {
	  this.a�o = a�o;
	  this.mes = mes;
	  this.dia = dia;
	  this.hora = hora;
	  this.minuto = minuto;
    }

	public static void main(String[] args) {
		Fecha f= new Fecha();
		System.out.println(f.a�o+":"+f.mes+":"+f.dia+":"+f.hora+":"+
		f.minuto);
	}

}
