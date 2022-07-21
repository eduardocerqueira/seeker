//date: 2022-07-21T16:56:56Z
//url: https://api.github.com/gists/f03e83c88f8ce67f25593bfb0d0536ec
//owner: https://api.github.com/users/eduardobentozup

package br.com.zup.edu.nutricionistas.controller;

import br.com.zup.edu.nutricionistas.repository.NutricionistaRepository;
import br.com.zup.edu.nutricionistas.util.MensagemDeErro;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultHandlers;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;

import java.time.LocalDate;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SpringBootTest
@AutoConfigureMockMvc
class CadastrarNutricionistaControllerTest {

	@Autowired
	private MockMvc mvc;

	@Autowired
	private NutricionistaRepository repository;

	@Autowired
	private ObjectMapper mapper;

	@BeforeEach
	public void before(){
		repository.deleteAll();
	}

	@Test
	@DisplayName("Should register a new nutritionist")
	void shouldRegisterANewNutritionist() throws Exception{

		NutricionistaRequest nutricionistaRequest = new NutricionistaRequest(
			"Eduardo",
			"email@email.com",
			"111111",
			"505.086.758-47",
			LocalDate.of(2001,01,01)
		);

		String payload = mapper.writeValueAsString(nutricionistaRequest);

		MockHttpServletRequestBuilder request = MockMvcRequestBuilders
			.post("/nutricionistas")
			.contentType(MediaType.APPLICATION_JSON)
			.content(payload)
			.header("Accept-Language", "pt-br");

		mvc.perform(request)
			.andDo(MockMvcResultHandlers.print())
			.andExpect(MockMvcResultMatchers.status().isCreated())
			.andExpect(MockMvcResultMatchers.redirectedUrlPattern("http://localhost/nutricionistas/*"));

	}

	@Test
	@DisplayName("Should validate the fields when they are not provided")
	void shouldValidateFieldsWhenTheyAreNotProvided() throws Exception {
		NutricionistaRequest nutricionistaRequest = new NutricionistaRequest();

		String payload = mapper.writeValueAsString(nutricionistaRequest);

		MockHttpServletRequestBuilder requestBuilder = MockMvcRequestBuilders
			.post("/nutricionistas")
			.contentType(MediaType.APPLICATION_JSON)
			.content(payload)
			.header("Accept-Language", "pt-br");

		String response = mvc.perform(requestBuilder)
			.andDo(MockMvcResultHandlers.print())
			.andExpect(MockMvcResultMatchers.status().isBadRequest())
			.andReturn()
			.getResponse()
			.getContentAsString(UTF_8);

		MensagemDeErro erros = mapper.readValue(response, MensagemDeErro.class);

		assertEquals(5, erros.getMensagens().size());
		assertThat(erros.getMensagens(), containsInAnyOrder(
			"O campo nome não deve estar em branco",
			"O campo email não deve estar em branco",
			"O campo CRN não deve estar em branco",
			"O campo cpf não deve estar em branco",
			"O campo dataNascimento não deve ser nulo"
		));

		assertEquals(0L, repository.count());
	}

	@Test
	@DisplayName("Should validate fields format")
	void shouldValidateFieldsFormat() throws Exception {

		NutricionistaRequest nutricionistaRequest = new NutricionistaRequest(
			"Eduardo",
			"emailInvalido.com",
			"111111",
			"123456789",
			LocalDate.of(2050,01,01)
		);

		String payload = mapper.writeValueAsString(nutricionistaRequest);

		MockHttpServletRequestBuilder requestBuilder = MockMvcRequestBuilders
			.post("/nutricionistas")
			.contentType(MediaType.APPLICATION_JSON)
			.content(payload)
			.header("Accept-Language", "pt-br");

		String response = mvc.perform(requestBuilder)
			.andDo(MockMvcResultHandlers.print())
			.andExpect(MockMvcResultMatchers.status().isBadRequest())
			.andReturn()
			.getResponse()
			.getContentAsString(UTF_8);

		MensagemDeErro erros = mapper.readValue(response, MensagemDeErro.class);

		assertEquals(3, erros.getMensagens().size());
		assertThat(erros.getMensagens(), containsInAnyOrder(
			"O campo email deve ser um endereço de e-mail bem formado",
			"O campo cpf número do registro de contribuinte individual brasileiro (CPF) inválido",
			"O campo dataNascimento deve ser uma data passada"
		));

		assertEquals(0L, repository.count());
	}
}