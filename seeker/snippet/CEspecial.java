//date: 2022-04-19T16:52:04Z
//url: https://api.github.com/gists/3555795d34fed4d2382c57036f8d03f5
//owner: https://api.github.com/users/DiogoLuizDeAquino

package Implementando_Conta_P;

public class CEspecial extends CCorrente {
    private int limite;

    public CEspecial(int num, double sal, String cli, int lim) {
        super(num, sal, cli);
        this.limite = lim;
}
    @Override
    public void debitar(double valor) {
        if (valor <= (super.get_saldo() + this.limite)) {
            super.set_saldo(super.get_saldo() - valor);
        } else {
            System.out.print("Saldo Insuficiente");
        }
    }
}
