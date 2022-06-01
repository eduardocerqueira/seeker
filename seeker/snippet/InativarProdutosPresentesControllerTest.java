//date: 2022-06-01T17:16:22Z
//url: https://api.github.com/gists/fb3d409e1b247f578c58a25fe25df8c6
//owner: https://api.github.com/users/thiagocavalcantezup

package br.com.zup.edu.marketplace.controller;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.TypeFactory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

import br.com.zup.edu.marketplace.model.Produto;
import br.com.zup.edu.marketplace.model.StatusProduto;
import br.com.zup.edu.marketplace.repository.ProdutoRepository;

@SpringBootTest
@AutoConfigureMockMvc(printOnlyOnFailure = false)
@ActiveProfiles("test")
public class InativarProdutosPresentesControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private ProdutoRepository prodRepository;

    @BeforeEach
    void setUp() {
        prodRepository.deleteAll();

        Produto prodAtivo = new Produto(
            "Guaraná Jesus", "Refrigerante", new BigDecimal("5.00"), StatusProduto.ATIVO,
            LocalDateTime.now()
        );
        prodRepository.save(prodAtivo);
    }

    @Test
    void deveRetornarUmaColecaoDeIdsDeProdutosInativados() throws Exception {
        // cenário (given)
        //
        Produto prodPendente1 = new Produto(
            "Café Santa Clara", "Café em pó", new BigDecimal("9.00")
        );
        Produto prodPendente2 = new Produto(
            "Biscoito Vitarella", "Biscoito sem recheio", new BigDecimal("9.00")
        );
        prodRepository.saveAll(List.of(prodPendente1, prodPendente2));

        MockHttpServletRequestBuilder request = post("/produtos/inativar").contentType(
            APPLICATION_JSON
        );

        // ação (when) e corretude (then)
        //
        String response = mockMvc.perform(request)
                                 .andExpect(status().isOk())
                                 .andReturn()
                                 .getResponse()
                                 .getContentAsString(UTF_8);

        TypeFactory typeFactory = objectMapper.getTypeFactory();
        List<Long> reponseIds = objectMapper.readValue(
            response, typeFactory.constructCollectionType(List.class, Long.class)
        );
        List<Long> savedIds = List.of(prodPendente1.getId(), prodPendente2.getId());

        assertThat(reponseIds).hasSize(2).containsAll(savedIds);

        savedIds.forEach((savedId) -> {
            Optional<Produto> optionalProdPendente = prodRepository.findById(savedId);

            assertTrue(optionalProdPendente.isPresent());
            assertEquals(StatusProduto.INATIVO, optionalProdPendente.get().getStatus());
        });
    }

    @Test
    void deveRetornarUmaColecaoVaziaCasoNaoExistamProdutosPendentes() throws Exception {
        // cenário (given)
        //
        MockHttpServletRequestBuilder request = post("/produtos/inativar").contentType(
            APPLICATION_JSON
        );

        // ação (when) e corretude (then)
        //
        String response = mockMvc.perform(request)
                                 .andExpect(status().isOk())
                                 .andReturn()
                                 .getResponse()
                                 .getContentAsString(UTF_8);

        TypeFactory typeFactory = objectMapper.getTypeFactory();
        List<Long> reponseIds = objectMapper.readValue(
            response, typeFactory.constructCollectionType(List.class, Long.class)
        );

        assertThat(reponseIds).isEmpty();
    }

}
