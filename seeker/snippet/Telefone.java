//date: 2024-01-31T16:56:10Z
//url: https://api.github.com/gists/9ac7869e06f3106f41107679f8efe973
//owner: https://api.github.com/users/DanielPuerari01

import java.util.Locale;
import java.util.Scanner;

public class Telefone {

	public static void main(String[] args) {
		Locale.setDefault(Locale.US);	
		Scanner sc = new Scanner(System.in);
		
		System.out.print("Informe a quantidade de clientes: ");
		int n = sc.nextInt();
		
		String[] nomes = new String[n];
		String[] telefones = new String[n];
		int[] tipos = new int[n];
		int[] minutos = new int[n];
		double[] valores = new double[n];
		
		double[][] contas = new double[3][2];
		
		for (int i=0; i<n; i++) {
			System.out.println("Dados do " + (i+1) + "o. cliente:");
			System.out.print("Nome: ");
			sc.nextLine();
			nomes[i] = sc.nextLine();
			System.out.print("Telefone: ");
			telefones[i] = sc.next();
			System.out.print("Tipo: ");
			tipos[i] = sc.nextInt();
			System.out.print("Minutos: ");
			minutos[i] = sc.nextInt();
			System.out.println();
		}
 		
		System.out.println("Informe o preco basico e excedente de cada tipo de conta:");
		
		for (int i=0; i<contas.length; i++) {
			System.out.println("Tipo " + i + ":");
			for (int j=0; j<contas[i].length; j++) {
				contas[i][j] = sc.nextDouble();
			}
		}
		
		for (int i=0; i<n; i++) {
			int tipoConta = tipos[i];
			double valor = contas[tipoConta][0];
			
			if (minutos[i] > 90) {
				valor += (minutos[i] - 90) * contas[tipoConta][1]; 
			}
			
			valores[i] = valor;
		}
		
		System.out.println();
		System.out.println("RELATÓRIO DE CLIENTES:");
		System.out.println();
		
		for (int i=0; i<n; i++) {
			System.out.println(nomes[i] + ", " + telefones[i] + ", Tipo " + tipos[i] + ", Minutos: " 
					+ minutos[i] + ", Conta = R$ " + String.format("%.2f", valores[i]));
		}
		
		sc.close();
		
	}

}
