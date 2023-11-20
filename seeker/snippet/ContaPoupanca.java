//date: 2023-11-20T16:32:02Z
//url: https://api.github.com/gists/db9930c6b867fd70d067b9b8b0612907
//owner: https://api.github.com/users/canaldojavao

package br.com.empresa.banco.conta;

import br.com.empresa.banco.titular.Titular;

import java.math.BigDecimal;

public class ContaPoupanca extends Conta {

    public ContaPoupanca(Titular titular, String agencia, String numero, BigDecimal saldoInicial) {
        super(titular, agencia, numero, saldoInicial);
    }

    @Override
    public void descontarTarifa() {
        this.saldo = this.saldo.subtract(new BigDecimal("0.2"));
    }

}
