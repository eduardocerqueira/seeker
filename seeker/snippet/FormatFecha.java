//date: 2022-09-01T17:08:18Z
//url: https://api.github.com/gists/d290ca9a4294c1a5519e96762ae8eb3c
//owner: https://api.github.com/users/akobashikawa

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class FormatFecha {
	public static void main(String[] args) throws ParseException {

		String[] fechas = { "01/12/2021",
			    "06/12/2021",
			    "08/12/2021",
			    "13/12/2021",
			    "15/12/2021",
			    "20/12/2021",
			    "22/12/2021",
			    "27/12/2021",
			    "29/12/2021" };
		String fechasAgrupadas = getFechasAgrupadas(fechas);

		System.out.println(fechasAgrupadas);
	}

	public static String getFechasAgrupadas(String[] fechas) throws ParseException {
		GsonBuilder builder = new GsonBuilder();
		builder.setPrettyPrinting();

		Gson gson = builder.create();
		
		ArrayList<GrupoDia> grupos = new ArrayList<GrupoDia>();
		grupos.add(new GrupoDia("lunes"));
		grupos.add(new GrupoDia("martes"));
		grupos.add(new GrupoDia("miercoles"));
		grupos.add(new GrupoDia("jueves"));
		grupos.add(new GrupoDia("viernes"));
		grupos.add(new GrupoDia("sabado"));
		grupos.add(new GrupoDia("domingo"));
		
		for (String fecha: fechas) {
			String dia = getDiaOf(fecha);
			for (GrupoDia grupo: grupos) {
				if (grupo.getDia() == dia) {
					grupo.addFecha(fecha);
					break;
				}
			}
		}

		String result = gson.toJson(grupos);
		return result;
	}
	
	public static String getDiaOf(String fecha) throws ParseException {
		String[] days = {"-", "domingo", "lunes", "martes", "miercoles", "jueves", "viernes", "sabado"};
		Date date = new SimpleDateFormat("dd/MM/yyyy").parse(fecha);
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		int dow = cal.get(Calendar.DAY_OF_WEEK);
		String result = days[dow];
		System.out.println(dow + result);
		return result;
	}
}

class GrupoDia {
	private String dia;
	private ArrayList<String> fechas = new ArrayList<String>();

	public GrupoDia() {
	}

	public GrupoDia(String dia) {
		this.dia = dia;
	}
	
	public String getDia() {
		return dia;
	}

	public void setDia(String dia) {
		this.dia = dia;
	}
	
	public ArrayList<String> getFechas() {
		return fechas;
	}

	public void setFechas(ArrayList<String> fechas) {
		this.fechas = fechas;
	}
	
	public void addFecha(String fecha) {
		this.fechas.add(fecha);
	}

	public String toString() {
		return "GrupoDia [ dia: " + dia + " ]";
	}
}