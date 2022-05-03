//date: 2022-05-03T16:59:16Z
//url: https://api.github.com/gists/10a5dd0dfb7901856999d1cedbfe8410
//owner: https://api.github.com/users/jordisilvazup

package br.com.zup.edu.avaliacoes.aluno;

import br.com.zup.edu.avaliacoes.compartilhado.MensagemDeErro;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static java.nio.charset.StandardCharsets.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.redirectedUrlPattern;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class CadastrarAvaliacaoAoAlunoControllerTest {
    @Autowired
    private ObjectMapper mapper;

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private AlunoRepository alunoRepository;

    @Autowired
    private AvaliacaoRepository avaliacaoRepository;

    private Aluno aluno;

    @BeforeEach
    void setUp() {
        avaliacaoRepository.deleteAll();
        alunoRepository.deleteAll();
        this.aluno = new Aluno("Jordi H M Silva", "jordi.silva@zup.com.br", "Aceleracao Academica Senior");
        alunoRepository.save(this.aluno);

    }

    @Test
    @DisplayName("deve cadastrar uma avaliacao ao Aluno")
    void test1() throws Exception {
        AvaliacaoRequest avaliacaoRequest = new AvaliacaoRequest(
                "Testando o Serviço de Cadastro de Avaliações",
                "Testando cadastros com relacionamentos Muitos para Um"
        );

        String payload = mapper.writeValueAsString(avaliacaoRequest);

        MockHttpServletRequestBuilder request = post("/alunos/{id}/avaliacoes", aluno.getId())
                .contentType(MediaType.APPLICATION_JSON)
                .content(payload);

        mockMvc.perform(request)
                .andExpect(
                        status().isCreated()
                )
                .andExpect(
                        redirectedUrlPattern("http://localhost/alunos/*/avaliacoes/*")
                );

        List<Avaliacao> avaliacoes = avaliacaoRepository.findAll();

        assertEquals(1, avaliacoes.size());
    }

    @Test
    @DisplayName("nao deve cadastrar uma avaliacao caso o Aluno nao exista")
    void test2() throws Exception {
        AvaliacaoRequest avaliacaoRequest = new AvaliacaoRequest(
                "Testando o Serviço de Cadastro de Avaliações",
                "Testando cadastros com relacionamentos Muitos para Um"
        );

        String payload = mapper.writeValueAsString(avaliacaoRequest);

        MockHttpServletRequestBuilder request = post("/alunos/{id}/avaliacoes", 1000)
                .contentType(MediaType.APPLICATION_JSON)
                .content(payload);

        mockMvc.perform(request)
                .andExpect(
                        status().isNotFound()
                );
    }

    @Test
    @DisplayName("nao deve cadastrar uma avaliacao com dados invalidos")
    void test3() throws Exception {
        AvaliacaoRequest avaliacaoRequest = new AvaliacaoRequest(null, null);

        String payload = mapper.writeValueAsString(avaliacaoRequest);

        MockHttpServletRequestBuilder request = post("/alunos/{id}/avaliacoes", aluno.getId())
                .contentType(MediaType.APPLICATION_JSON)
                .header("Accept-Language","pt-br")
                .content(payload);

        String payloadResponse = mockMvc.perform(request)
                .andExpect(
                        status().isBadRequest()
                )
                .andReturn()
                .getResponse()
                .getContentAsString(UTF_8);

        MensagemDeErro mensagemDeErro = mapper.readValue(payloadResponse, MensagemDeErro.class);

        assertEquals(2, mensagemDeErro.getMensagens().size());
        assertThat(mensagemDeErro.getMensagens(), containsInAnyOrder(
                "O campo titulo não deve estar em branco",
                "O campo avaliacaoReferente não deve estar em branco"
                )
        );

    }
}