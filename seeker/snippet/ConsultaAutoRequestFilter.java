//date: 2024-12-17T16:58:19Z
//url: https://api.github.com/gists/7aa5a35cb8941c06fe3c433107da5096
//owner: https://api.github.com/users/antonino3g

package br.com.qg.gatewaybiro.consultaauto;

import br.com.qg.gatewaybiro.executorconsulta.FiltroAutomotiva;
import br.com.qg.gatewaybiro.executorconsulta.FiltroDocumento;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serial;
import java.io.Serializable;

public class ConsultaAutoRequestFilter implements FiltroAutomotiva, FiltroDocumento, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @JsonProperty
    private String chassi;
    @JsonProperty
    private String placa;

    @Override
    public String getChassi() {
        return chassi;
    }

    public void setChassi(String chassi) {
        this.chassi = chassi;
    }

    @Override
    public String getPlaca() {
        return placa;
    }

    public void setPlaca(String placa) {
        this.placa = placa;
    }

    @Override
    public String getDocumento() {
        return null;
    }
}
