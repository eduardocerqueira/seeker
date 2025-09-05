//date: 2025-09-05T16:43:19Z
//url: https://api.github.com/gists/10f9c87805f041e407b4985ea33e4531
//owner: https://api.github.com/users/shinoyori

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

import java.util.ArrayList;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class RelatorioDeFuncionariosTest {

 private RelatorioDeFuncionarios relatorio;
 private FuncionarioDAO funcDao;
 private ReceitaFederal rf;

 @BeforeEach
 public void setUp() {
  funcDao = mock(FuncionarioDAO.class);
  rf = mock(ReceitaFederal.class);
  relatorio = new RelatorioDeFuncionarios(funcDao);
  relatorio.setRf(rf);
 }

 @Test
 public void testNenhumFuncionarioComCPFBloqueado() {
  ArrayList<Funcionario> funcionarios = new ArrayList<>();
  funcionarios.add(new Funcionario(1, "Funcionario1", "123456789-00"));
  funcionarios.add(new Funcionario(2, "Funcionario2", "987654321-00"));

  when(funcDao.getFuncionariosBy("tecnico")).thenReturn(funcionarios);
  when(rf.isCPFBloqueado(anyString())).thenReturn(false);

  int resultado = relatorio.getFuncComCPFBloqueado("tecnico");

  assertEquals(0, resultado);
  verify(funcDao).getFuncionariosBy("tecnico");
  verify(rf, times(2)).isCPFBloqueado(anyString());
 }

 @Test
 public void testUmFuncionarioComCPFBloqueado() {
  ArrayList<Funcionario> funcionarios = new ArrayList<>();
  funcionarios.add(new Funcionario(1, "Funcionario1", "123456789-00"));
  when(funcDao.getFuncionariosBy("analista")).thenReturn(funcionarios);
  when(rf.isCPFBloqueado("123456789-00")).thenReturn(true);

  int resultado = relatorio.getFuncComCPFBloqueado("analista");

  assertEquals(1, resultado);
  verify(funcDao).getFuncionariosBy("analista");
  verify(rf).isCPFBloqueado("123456789-00");
 }

 @Test
 public void testDoisFuncionariosComCPFBloqueadoEmUmaListaDeQuatro() {
  ArrayList<Funcionario> funcionarios = new ArrayList<>();
  funcionarios.add(new Funcionario(1, "Funcionario1", "123456789-00"));
  funcionarios.add(new Funcionario(2, "Funcionario2", "111222333-44"));
  funcionarios.add(new Funcionario(3, "Funcionario3", "654321987-23"));
  funcionarios.add(new Funcionario(4, "Funcionario4", "098876654-99"));

  when(funcDao.getFuncionariosBy("gerente")).thenReturn(funcionarios);
  when(rf.isCPFBloqueado("123456789-00")).thenReturn(false);
  when(rf.isCPFBloqueado("111222333-44")).thenReturn(true);
  when(rf.isCPFBloqueado("654321987-23")).thenReturn(false);
  when(rf.isCPFBloqueado("098876654-99")).thenReturn(true);

  int resultado = relatorio.getFuncComCPFBloqueado("gerente");

  assertEquals(2, resultado);
  verify(funcDao).getFuncionariosBy("gerente");
  verify(rf).isCPFBloqueado("123456789-00");
  verify(rf).isCPFBloqueado("111222333-44");
  verify(rf).isCPFBloqueado("654321987-23");
  verify(rf).isCPFBloqueado("098876654-99");
 }
}