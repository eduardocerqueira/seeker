//date: 2024-02-12T17:01:55Z
//url: https://api.github.com/gists/18b8fdb29e36b1f811186ec3ee7237ea
//owner: https://api.github.com/users/SLB-1986

import java.util.*;

public class XsiO {

	static ArrayList<Integer> pozitiiJucator = new ArrayList<Integer>();
	static ArrayList<Integer> pozitiiComputer = new ArrayList<Integer>();

	public static void main(String[] args) {
//cream tabla de joc pentru X si 0
		char [][] loculDeJoaca = {
				{' ', '|', ' ', '|', ' '},
				{'-', '+', '-', '+', '-'},
				{' ', '|', ' ', '|', ' '},
				{'-', '+', '-', '+', '-'},
				{' ', '|', ' ', '|', ' '}};

		listeazaLoculDeJoaca(loculDeJoaca); // apelam metoda listeaza locul de joaca

		while (true) {

			Scanner sc = new Scanner(System.in);
			System.out.println("Intro locul unde doresti sa pozitionezi X (de la 1 la 9): ");

			int pozitieJucator = sc.nextInt(); // definim un int ca pozitie jucator sa stim ca acesta este cel introdus de utilizator/jucator
			while(pozitiiJucator.contains(pozitieJucator) || pozitiiComputer.contains(pozitieJucator)){
				System.out.println("Pozitie ocupata! Introdu o pozitie corecta din cele libere: ");
				pozitieJucator = sc.nextInt();
			}

			puneX(loculDeJoaca, pozitieJucator, "Jucator");

			String rezultat = verificaCastigator();
			if(rezultat.length() > 0){
				System.out.println(rezultat);
				break;
			}

			Random aleatoriu = new Random(); // apelam metoda aleatorie numita random pentru a solicita sistemului sa puna aleatoriu un caracter

			int pozitieComputer = aleatoriu.nextInt(9)+1;
			while(pozitiiJucator.contains(pozitieComputer) || pozitiiComputer.contains(pozitieJucator)){
				pozitieComputer = aleatoriu.nextInt(9)+1;
			}

			puneX(loculDeJoaca, pozitieComputer, "Calculator");

			listeazaLoculDeJoaca(loculDeJoaca); // reapelam metoda listeaza locul de joaca pentru a putea introduce modificarile in tabela de joc

			String result = verificaCastigator();
			if(result.length() > 0){
				System.out.println(result);
				break;
			}
		}
	}

	public static void listeazaLoculDeJoaca (char [][] loculDeJoaca){ // am creat o metoda numita listeaza locul de joaca, care contine tabla de joc
		for(char[] coloana : loculDeJoaca){
			for(char c : coloana){
			System.out.print(c);
			}
		System.out.println();
		}
	}

	public static void puneX(char[][] loculDeJoaca, int pozitia, String utilizator){

		char simbol = ' ';
		if(utilizator.equals("Jucator")){
			simbol = 'X';
			pozitiiJucator.add(pozitia);
		}else if(utilizator.equals("Calculator")){
			simbol = '0';
			pozitiiComputer.add(pozitia);
		}

		switch (pozitia){
			case 1: loculDeJoaca[0][0] = simbol;
				break;
			case 2: loculDeJoaca[0][2] = simbol;
				break;
			case 3: loculDeJoaca[0][4] = simbol;
				break;
			case 4: loculDeJoaca[2][0] = simbol;
				break;
			case 5: loculDeJoaca[2][2] = simbol;
				break;
			case 6: loculDeJoaca[2][4] = simbol;
				break;
			case 7: loculDeJoaca[4][0] = simbol;
				break;
			case 8: loculDeJoaca[4][2] = simbol;
				break;
			case 9: loculDeJoaca[4][4] = simbol;
				break;
			default:
				break;
		}
	}

	public static String verificaCastigator(){ // dejinim conditia de joc si conditiile de castig
		List linieSus = Arrays.asList(1,2,3);
		List linieMijloc = Arrays.asList(4,5,6);
		List linieJos = Arrays.asList(7,8,9);
		List coloanaStanga = Arrays.asList(1,4,7);
		List coloanaMijloc = Arrays.asList(2,5,8);
		List coloanaDreapta = Arrays.asList(3,6,9);
		List diagonalaStanga = Arrays.asList(1,5,9);
		List diagonalDreapta = Arrays.asList(3,5,7);

		List<List> conditiaDeCastig = new ArrayList<List>();
		conditiaDeCastig.add(linieSus);
		conditiaDeCastig.add(linieMijloc);
		conditiaDeCastig.add(coloanaStanga);
		conditiaDeCastig.add(coloanaDreapta);
		conditiaDeCastig.add(coloanaMijloc);
		conditiaDeCastig.add(diagonalDreapta);
		conditiaDeCastig.add(diagonalaStanga);
		conditiaDeCastig.add(linieJos);

		for(List contidiileDeCastig : conditiaDeCastig){
			if(pozitiiJucator.containsAll(contidiileDeCastig)){
				return "Felicitari ai castigat!";
			}else if(pozitiiComputer.containsAll(contidiileDeCastig)){
				return "Computerul a castigat!";
			}else if(pozitiiJucator.size() + pozitiiComputer.size() == 9){
				return "Remiza!";
			}

		}

		return ""; /** fiindca am intoarcem un String gol aici, am creat aceasta referinta pentru rezultat:
		 if(rezultat.length() > 0){
		 System.out.println(rezultat);
		 break;
		 }*/
	}

}