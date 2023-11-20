//date: 2023-11-20T16:32:02Z
//url: https://api.github.com/gists/db9930c6b867fd70d067b9b8b0612907
//owner: https://api.github.com/users/canaldojavao

package br.com.empresa.banco;

import br.com.empresa.banco.conta.ContaCorrente;
import br.com.empresa.banco.titular.Titular;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Principal {

    public static void main(String[] args) {
        var fulano = new Titular("Fulano da Silva", "000.000.000-00", "fulano@email.com");
        var contaDoFulano = new ContaCorrente(fulano, "0001", "123456-0", new BigDecimal("25.0"));

        System.out.println("Dados bancários:");
        System.out.println("titular: " +contaDoFulano.getTitular().getNome());
        System.out.println("Agência: " +contaDoFulano.getAgencia());
        System.out.println("Conta: " +contaDoFulano.getNumero());
        System.out.println("Saldo atual: " +contaDoFulano.getSaldo());

        //Tentando sacar 10 reais
        var sacou = contaDoFulano.sacar(new BigDecimal("10.0"));
        if (sacou) {
            System.out.println("Saque realizado com sucesso!");
            System.out.println("Saldo atualizado: " +contaDoFulano.getSaldo());
        }


        System.out.println(contaDoFulano);
        //FQN -> Full Qualified Name


        var conta2 = new ContaCorrente(fulano, "0002", "123456-0", new BigDecimal("25.0"));
        conta2.sacar(new BigDecimal("10.0"));

        System.out.println(conta2);

        System.out.println(contaDoFulano.equals(conta2));


//        var valor1 = new BigDecimal("3.3");
//        var valor2 = new BigDecimal("3.3");
//        var valor3 = new BigDecimal("3.3");
//
//        System.out.println(valor1.add(valor2).add(valor3));
//        System.out.println(valor1);

        var valor4 = new BigDecimal("10");
        var valor5 = new BigDecimal("3");

        var resultado = valor4.divide(valor5, RoundingMode.HALF_UP).setScale(2);

        System.out.println(resultado);


    }

}
