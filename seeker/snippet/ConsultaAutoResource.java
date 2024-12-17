//date: 2024-12-17T16:56:10Z
//url: https://api.github.com/gists/78b96d809e96d7c19c9f728c7cab5e46
//owner: https://api.github.com/users/antonino3g

package br.com.qg.gatewaybiro.consultaauto;

import br.com.qg.gatewaybiro.consulta.OrigemConsulta;
import br.com.qg.gatewaybiro.executorconsulta.OpcaoConsulta;
import br.com.qg.gatewaybiro.executorconsulta.ProcessadorConsultaService;
import br.com.qg.gatewaybiro.infra.IPUtil;
import br.com.qg.gatewaybiro.rest.Secured;
import br.com.qg.gatewaybiro.rest.UserLogadoRS;
import br.com.qg.gatewaybiro.usuario.Usuario;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import jakarta.inject.Inject;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

@Path("/gatewaybiro")
public class ConsultaAutoResource {

    @POST
    @Secured(permission = "AUTOMOTIVA_HISTORICOPROPRIETARIO")
    @Path("/automotivahistoricoproprietario")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    @SecurityRequirement(name = "AuThorization")
    @Operation(description = "Consultar histórico de Proprietário", hidden = false, security = {
            @SecurityRequirement(name = "Bearer Authentication")})
    public Response consultaHistoricoProprietario(
            ConsultaAutoRequestFilter filter,
            @Context HttpServletRequest servletRequest) {
        return Response.ok(
                processadorConsultaService.processarConsulta(
                        OpcaoConsulta.CONSULTAAUTO_HISTORICOPROPRIETARIO,
                        filter,
                        userLogadoRS.isCache(),
                        IPUtil.getIpAddr(servletRequest),
                        OrigemConsulta.API)
        ).build();
    }
    @Inject
    @UserLogadoRS
    private Usuario userLogadoRS;

    @Inject
    private ProcessadorConsultaService processadorConsultaService;
}
