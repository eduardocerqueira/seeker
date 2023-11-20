//date: 2023-11-20T16:32:02Z
//url: https://api.github.com/gists/db9930c6b867fd70d067b9b8b0612907
//owner: https://api.github.com/users/canaldojavao

package br.com.empresa.banco.conta;

import br.com.empresa.banco.titular.Titular;

import java.math.BigDecimal;
import java.util.Objects;

public abstract class Conta {

    private String agencia;
    private String numero;
    protected BigDecimal saldo;
    private Titular titular;

    public Conta(Titular titular, String agencia, String numero, BigDecimal saldoInicial) {
        this.titular = titular;
        this.agencia = agencia;
        this.numero = numero;
        this.saldo = BigDecimal.ZERO;
        depositar(saldoInicial);
    }

    public boolean sacar(BigDecimal valor) {
        if (valor.equals(BigDecimal.ZERO)) {
            return false;
        }

        if(saldo.compareTo(valor) >= 0) {
            saldo = saldo.subtract(valor);
            return true;
        }

        return false;
    }

    public void depositar(BigDecimal valor) {
        if (valor.equals(BigDecimal.ZERO)) {
            return;
        }

        this.saldo = this.saldo.add(valor);
    }

    public abstract void descontarTarifa();

    @Override
    public String toString() {
        return "Conta{" +
                "agencia='" + agencia + '\'' +
                ", numero='" + numero + '\'' +
                ", saldo=" + saldo +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Conta conta = (Conta) o;
        return Objects.equals(agencia, conta.agencia) && Objects.equals(numero, conta.numero);
    }

    @Override
    public int hashCode() {
        return Objects.hash(agencia, numero);
    }

    public String getAgencia() {
        return agencia;
    }
    public String getNumero() {
        return numero;
    }
    public BigDecimal getSaldo() {
        return saldo;
    }
    public Titular getTitular() {
        return titular;
    }

}
