//date: 2022-06-03T17:05:46Z
//url: https://api.github.com/gists/702ef1542e75f749a7b6cbe6d7a1826f
//owner: https://api.github.com/users/josearaujozup

package br.com.zup.edu.petmanager.controller;

import br.com.zup.edu.petmanager.controller.response.PetResponse;
import br.com.zup.edu.petmanager.model.Pet;
import br.com.zup.edu.petmanager.model.TipoPet;
import br.com.zup.edu.petmanager.repository.PetRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;
import org.springframework.web.server.ResponseStatusException;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@AutoConfigureMockMvc(printOnlyOnFailure = false)
@ActiveProfiles("test")
class DetalharPetControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper mapper;

    @Autowired
    private PetRepository petRepository;

    @BeforeEach
    void setUp(){
        petRepository.deleteAll();
    }

    @Test
    @DisplayName("Deve detalhar os dados de pet")
    void test1() throws Exception {
        Pet pet = new Pet("Bruno", "Vira-lata", TipoPet.GATO, LocalDate.now().minusMonths(5));
        petRepository.save(pet);

        MockHttpServletRequestBuilder request = MockMvcRequestBuilders.get("/pets/{id}", pet.getId());

        String payload = mockMvc.perform(request)
                .andExpect(
                        MockMvcResultMatchers.status().isOk()
                )
                .andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8);

        PetResponse response = mapper.readValue(payload, PetResponse.class);

        assertNotNull(response);
        assertThat(response)
                .extracting("nome","raca","tipo")
                .contains(pet.getNome(),pet.getRaca(),pet.getTipo().name().toLowerCase());
    }

    @Test
    @DisplayName("Não deve detalhar um pet não cadastrado")
    void test2() throws Exception {
        MockHttpServletRequestBuilder request = MockMvcRequestBuilders.get("/pets/{id}", Integer.MAX_VALUE);

        Exception resolvedException = mockMvc.perform(request)
                .andExpect(
                        MockMvcResultMatchers.status().isNotFound()
                )
                .andReturn()
                .getResolvedException();

        assertNotNull(resolvedException);
        assertEquals(ResponseStatusException.class,resolvedException.getClass());
        assertEquals("Pet não cadastrado.",((ResponseStatusException) resolvedException).getReason());
    }

}