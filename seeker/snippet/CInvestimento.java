//date: 2022-04-19T16:52:04Z
//url: https://api.github.com/gists/3555795d34fed4d2382c57036f8d03f5
//owner: https://api.github.com/users/DiogoLuizDeAquino

package Implementando_Conta_P;

import java.util.*;

public class CInvestimento extends CCorrente {
    private int dia_investimento;
    private int periodo;
    public CInvestimento(int num, float sal, String cli, int dinv, int per) {
        super(num, sal, cli);
        this.dia_investimento = dinv;
        this.periodo = per;
    }
    public void atualizar_saldo(){
        Calendar dataAtual = Calendar.getInstance();
        int dia = dataAtual.get(Calendar.DAY_OF_MONTH);
        System.out.println(dia);
        if (dia == (this.dia_investimento+periodo)){
            super.set_saldo((float) (super.get_saldo() * 1.20));
        }
        else {
            System.out.println("Ainda n dia do investimento");
        }
    }
}
