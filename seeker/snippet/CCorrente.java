//date: 2022-04-19T16:52:04Z
//url: https://api.github.com/gists/3555795d34fed4d2382c57036f8d03f5
//owner: https://api.github.com/users/DiogoLuizDeAquino

package Implementando_Conta_P;

public class CCorrente {

    private int numero;
    private double saldo;
    private String cliente;

    public CCorrente(int num, double sal, String cli){
        this.numero = num;
        this.saldo = sal;
        this.cliente = cli;
    }

    public void creditar(double valor){
        this.saldo = this.saldo+valor;
    }

    public void debitar(double valor) {
        if (valor <= this.saldo) {
            this.saldo = this.saldo - valor;
        } else {
            System.out.println("Saldo Insuficiente");
        }
    }public double get_saldo(){
        return  this.saldo;
    }

    public void set_saldo(double saldo) {
        this.saldo = saldo;

    }
}
