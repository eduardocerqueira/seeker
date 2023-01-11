//date: 2023-01-11T16:50:48Z
//url: https://api.github.com/gists/3b8755273d68ae173c4dce28add383ae
//owner: https://api.github.com/users/jose-junior-maker

import java.util.Scanner;

/**
 *
 * @author jose
 */
public class Atleta {

    public static void main(String[] args) {
        
        int n = 0, qtd_m = 0, qtd_f = 0;
        Double peso, altura, total_peso = 0.0, maior = 0.0, total_altura = 0.0, media_altura;
        String nome = null, sexo = null, nome_alto = "";
        
        
        Scanner sc = new Scanner(System.in);
        System.out.print("Qual a quantidade de atletas? ");
        n = sc.nextInt();
        
        for (int i = 1; i <= n; i++){
            
            System.out.printf("Digite os dados do atleta numero %d:\n", i);
            sc.nextLine();
            
            System.out.printf("Informe um nome: ");
            nome = sc.nextLine();
            System.out.printf("Informe um sexo: ");
            sexo = sc.nextLine();
            
            while (!"F".equals(sexo) && !"M".equals(sexo)){
               System.out.printf("Valor invalido! Favor digitar F ou M: ");
               sexo = sc.nextLine();
            }
            
            System.out.printf("Informe uma altura: ");
            altura = sc.nextDouble();
            
            
            while (altura <= 0){
                System.out.printf("Valor invalido! Favor digitar um valor positivo: ");
                altura = sc.nextDouble();
            }
            
            System.out.printf("Informe um peso: ");
            peso = sc.nextDouble();
            
            while (peso <= 0){
                System.out.printf("Valor invalido! Favor digitar um valor positivo: ");
                peso = sc.nextDouble();
            }
            
            total_peso = total_peso + peso;
            
            
            if ("F".equals(sexo)){
                qtd_f++;
                total_altura = total_altura + altura;
            }
            
            if ("M".equals(sexo)){
                qtd_m++;
            }
            
            if (altura > maior){
                maior = altura;
                nome_alto = nome;
            }
            
        }
        
        double media = total_peso/n;
        double porcentagem = (qtd_m * 100)/n;
        
        
                
        System.out.println();
        System.out.println("RELATÓRIO: ");
        System.out.printf("Peso médio dos atletas: %.2f\n", media);
        System.out.printf("Atleta mais alto: %s\n", nome_alto);
        System.out.printf("Porcentagem de homens: %.1f %%%n", porcentagem);
        
        if (qtd_f == 0){
            System.out.print("Não há mulheres cadastradas");
        }else {
            media_altura = total_altura/qtd_f;
            System.out.printf("Altura média das mulheres: %.2f", media_altura);
        }
    }
}