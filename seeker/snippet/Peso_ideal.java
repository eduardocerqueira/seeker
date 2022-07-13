//date: 2022-07-13T17:05:42Z
//url: https://api.github.com/gists/df1ce25567adc2bf4c0e8d5279d59624
//owner: https://api.github.com/users/EMoyaP

import javax.swing.*;

public class Peso_ideal {
    public static void main(String[] args) {
        String genero = "";
        do {
            genero = JOptionPane.showInputDialog("Ingrese tu genero:   (H/M)");
        } while (genero.equalsIgnoreCase("H") == false && genero.equalsIgnoreCase("M") == false);

        double altura = (double) Integer.parseInt(JOptionPane.showInputDialog("Ingrese su altura en cm "));
        double peso = (double) Integer.parseInt(JOptionPane.showInputDialog("Ingrese su peso en kg "));
        double IMC = peso / ((altura / 100) * (altura / 100));
        double pesoIdeal = 0;
        String resultadoIMC = "";

        if (genero.equalsIgnoreCase("H")) {
            pesoIdeal = (50 + ((double) altura - 150) * 0.92);
        } else if (genero.equalsIgnoreCase("M")) {
            pesoIdeal = (45.5 + ((double) altura - 150) * 0.92);
        }
        if (IMC < 18.5) {
            resultadoIMC = "Peso bajo";
        } else if (IMC >= 18.5 && IMC <= 24.9) {
            resultadoIMC = "Peso normal";
        } else if (IMC >= 25 && IMC <= 29.9) {
            resultadoIMC = "Sobrepeso";
        } else if (IMC >= 30 && IMC <= 34.9) {
            resultadoIMC = "Obesidad grado I";
        } else if (IMC >= 35 && IMC <= 39.9) {
            resultadoIMC = "Obesidad grado II";
        } else if (IMC >= 40) {
            resultadoIMC = "Obesidad grado III";
        }
        JOptionPane.showMessageDialog(null, "Su peso ideal es: " + pesoIdeal + " kg\n" + "Su IMC es: " + String.format("%1.2f",IMC) + "\n" + "Su IMC es: " + resultadoIMC);
    }
}
