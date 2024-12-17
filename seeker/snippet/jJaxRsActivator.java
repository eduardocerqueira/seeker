//date: 2024-12-17T17:01:55Z
//url: https://api.github.com/gists/9f119e412c1af86907f46903ffbd25cf
//owner: https://api.github.com/users/antonino3g

package br.com.qg.gatewaybiro.rest;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import br.com.qg.gatewaybiro.apideconsulta.api.ApiDeConsultaResource;
import br.com.qg.gatewaybiro.checkprice.api.CheckpriceResource;
import br.com.qg.gatewaybiro.cliente.ClienteResource;
import br.com.qg.gatewaybiro.consulta.ConsultaResource;
import br.com.qg.gatewaybiro.consultaautocorp.api.ConsultaAutocorpResource;
import br.com.qg.gatewaybiro.consultapositivo.ConsultaPositivoResource;
import br.com.qg.gatewaybiro.credauto.api.CredautoResource;
import br.com.qg.gatewaybiro.equifax.ConsultaEquifaxResource;
import br.com.qg.gatewaybiro.estado.EstadoResource;
import br.com.qg.gatewaybiro.fatura.FaturaResource;
import br.com.qg.gatewaybiro.faturamento.FaturamentoResource;
import br.com.qg.gatewaybiro.franquia.FranquiaResource;
import br.com.qg.gatewaybiro.infosimples.api.InfoSimplesResource;
import br.com.qg.gatewaybiro.karfex.api.KarfexResource;
import br.com.qg.gatewaybiro.leilao.ConsultaLeilaoResource;
import br.com.qg.gatewaybiro.leilao.LeilaoResource;
import br.com.qg.gatewaybiro.mtix.api.MtixResource;
import br.com.qg.gatewaybiro.municipio.MunicipioResource;
import br.com.qg.gatewaybiro.nexo.api.NexoResource;
import br.com.qg.gatewaybiro.parametroaplicacao.ParametroAplicacaoResource;
import br.com.qg.gatewaybiro.permissao.PermissaoResource;
import br.com.qg.gatewaybiro.pessoa.PessoaResource;
import br.com.qg.gatewaybiro.pessoajuridica.PessoaJuridicaResource;
import br.com.qg.gatewaybiro.placafipeapi.api.PlacaFipeResource;
import br.com.qg.gatewaybiro.plano.PlanoResource;
import br.com.qg.gatewaybiro.protesto.ConsultaProtestoREST;
import br.com.qg.gatewaybiro.registro.RegistroResource;
import br.com.qg.gatewaybiro.relatorio.RelatorioResource;
import br.com.qg.gatewaybiro.servico.ServicoResource;
import br.com.qg.gatewaybiro.sinistro.ConsultaSinistroResource;
import br.com.qg.gatewaybiro.sophus.ConsultaSophusResource;
import br.com.qg.gatewaybiro.ultracheck.UltracheckResource;
import br.com.qg.gatewaybiro.usuario.UsuarioResource;
import br.com.qg.gatewaybiro.veiculo.VeiculoResource;
import br.com.qg.gatewaybiro.veiculovenda.VeiculoVendaResource;
import br.com.qg.gatewaybiro.web.ConsultaWebResource;
import io.swagger.v3.jaxrs2.integration.resources.AcceptHeaderOpenApiResource;
import io.swagger.v3.jaxrs2.integration.resources.OpenApiResource;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Contact;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.info.License;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import jakarta.ws.rs.ApplicationPath;
import jakarta.ws.rs.core.Application;

@OpenAPIDefinition(
		  info =@Info(
		    title = "CREDIT BUREAU SYSTEM",
		    version = "1.0",
		    contact = @Contact(
		      name = "Deoclides Quevedo", email = "deoclides.quevedo@gmail.com", url = "https://b1.toolsdata.com.br/swagger"
		    ),
		    license = @License(
		      name = "Apache 2.0", url = "https://www.apache.org/licenses/LICENSE-2.0"
		    ),
		    termsOfService = "www.google.com",
		    description = "API CREDIT BUREAU SYSTEM"
		  )
		)
@SecurityScheme(
		  name = "Bearer Authentication",
		  type = SecuritySchemeType.HTTP,
		  bearerFormat = "JWT",
		  scheme = "bearer"
		)
@ApplicationPath("/api")
public class JaxRsActivator extends Application {
   /* class body intentionally left blank */
	 @Override
	    public Set<Class<?>> getClasses() {
	        return Stream.of(AuthenticationREST.class, 
	        		ConsultaREST.class, 
	        		AuthenticationFilter.class, 
	        		CorsFilter.class, 
	        		ConsultaProtestoREST.class, 
	        		ConsultaEquifaxResource.class,
	        		PessoaResource.class,
	        		PessoaJuridicaResource.class,
	        		RelatorioResource.class,	
	        		FranquiaResource.class,
	        		LeilaoResource.class,
	        		VeiculoResource.class,
	        		ClienteResource.class,
	        		RegistroResource.class,
	        		ConsultaWebResource.class,
	        		ConsultaResource.class,
	        		PermissaoResource.class,
	        		UsuarioResource.class,
	        		ServicoResource.class,
	        		FaturaResource.class,
	        		PlanoResource.class,
	        		FaturamentoResource.class,
	        		VeiculoVendaResource.class,
	        		CheckpriceResource.class,
	        		EstadoResource.class,
	        		MunicipioResource.class,
	        		ConsultaSophusResource.class,
	        		ConsultaPositivoResource.class,
	        		OpenApiResource.class,
	        		ParametroAplicacaoResource.class,
	        		UltracheckResource.class,
					ConsultaSinistroResource.class,
	        		AcceptHeaderOpenApiResource.class,
	        		ConsultaLeilaoResource.class,
					ConsultaAutocorpResource.class,
					ApiDeConsultaResource.class,
					NexoResource.class,
					CredautoResource.class,
					KarfexResource.class,
					MtixResource.class,
					PlacaFipeResource.class,
					ConsultaAutocorpResource.class,
					InfoSimplesResource.class).collect(Collectors.toSet());
	    }
}
