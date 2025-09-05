//date: 2025-09-05T16:43:19Z
//url: https://api.github.com/gists/10f9c87805f041e407b4985ea33e4531
//owner: https://api.github.com/users/shinoyori

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ClassificadorTest {
    private Classificador classificador;

    @BeforeEach
    void init() {
        classificador = new Classificador();
    }

    @Test
    @DisplayName("Lança exceção para idade negativa ou inválida")
    void deveLancarExcecaoQuandoIdadeInvalida() {
        Pessoa pessoaInvalida = new Pessoa("Invalido", -1);
        assertThrows(IllegalArgumentException.class, 
            () -> classificador.definirFaixaEtaria(pessoaInvalida)
        );
    }

    @Test
    @DisplayName("Classifica corretamente uma criança")
    void classificaCriancaCorretamente() {
        Pessoa ana = new Pessoa("Ana", 10);
        String faixa = classificador.definirFaixaEtaria(ana);
        assertEquals("Ana eh crianca", faixa);
    }

    @Test
    @DisplayName("Classifica corretamente um adolescente")
    void classificaAdolescenteCorretamente() {
        Pessoa beto = new Pessoa("Beto", 18);
        String faixa = classificador.definirFaixaEtaria(beto);
        assertEquals("Beto eh adolescente", faixa);
    }

    @Test
    @DisplayName("Classifica corretamente um adulto")
    void classificaAdultoCorretamente() {
        Pessoa carla = new Pessoa("Carla", 35);
        String faixa = classificador.definirFaixaEtaria(carla);
        assertEquals("Carla eh adulto", faixa);
    }

    @Test
    @DisplayName("Classifica corretamente um idoso")
    void classificaIdosoCorretamente() {
        Pessoa dario = new Pessoa("Dario", 60);
        String faixa = classificador.definirFaixaEtaria(dario);
        assertEquals("Dario eh idoso", faixa);
    }
}
