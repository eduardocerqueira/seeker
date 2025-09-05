//date: 2025-09-05T16:43:19Z
//url: https://api.github.com/gists/10f9c87805f041e407b4985ea33e4531
//owner: https://api.github.com/users/shinoyori

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class IMCCalculadoraTest {

 IMCCalculadora calculadora;

 @BeforeEach
 public void setup() {
  calculadora = new IMCCalculadora();
 }

 @Test
 public void testClassificacaoAbaixoDoPeso() {
  Pessoa pessoa = new Pessoa("Joao", 50.0, 1.80);
  IMCStatus status = calculadora.calcular(pessoa);
  assertEquals(50.0 / (1.80 * 1.80), status.getImc(), 0.001);
  assertEquals("abaixo do peso", status.getClassificacao());
 }

 @Test
 public void testClassificacaoNormal() {
  Pessoa pessoa = new Pessoa("Maria", 70.0, 1.75);
  IMCStatus status = calculadora.calcular(pessoa);
  assertEquals(70.0 / (1.75 * 1.75), status.getImc(), 0.001);
  assertEquals("normal", status.getClassificacao());
 }

 @Test
 public void testClassificacaoAcimaDoPeso() {
  Pessoa pessoa = new Pessoa("Jose", 85.0, 1.70);
  IMCStatus status = calculadora.calcular(pessoa);
  assertEquals(85.0 / (1.70 * 1.70), status.getImc(), 0.001);
  assertEquals("acima do peso", status.getClassificacao());
 }

 @Test
 public void testClassificacaoObeso() {
  Pessoa pessoa = new Pessoa("Ana", 100.0, 1.65);
  IMCStatus status = calculadora.calcular(pessoa);
  assertEquals(100.0 / (1.65 * 1.65), status.getImc(), 0.001);
  assertEquals("obeso", status.getClassificacao());
 }

 @Test
 public void testLancaExcecaoComValoresInvalidos() {
  Pessoa pessoa1 = new Pessoa("Invalida", 0, 1.70);
  assertThrows(IllegalArgumentException.class, () -> calculadora.calcular(pessoa1));

  Pessoa pessoa2 = new Pessoa("Invalida", 70.0, 0);
  assertThrows(IllegalArgumentException.class, () -> calculadora.calcular(pessoa2));

  Pessoa pessoa3 = new Pessoa("Invalida", -10, 1.70);
  assertThrows(IllegalArgumentException.class, () -> calculadora.calcular(pessoa3));

  Pessoa pessoa4 = new Pessoa("Invalida", 70.0, -1.70);
  assertThrows(IllegalArgumentException.class, () -> calculadora.calcular(pessoa4));
 }
}