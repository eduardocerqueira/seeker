#date: 2022-02-07T16:58:59Z
#url: https://api.github.com/gists/82c32e1bd50cb5ec95b407ac1fcba20f
#owner: https://api.github.com/users/Arduinobymyself

'''
Esta é a classe de testes automatizados dos ordenadores
do curso de Python da Coursera-2
Chama os módulos ordenador e contatempos
deveser salvo como test_ordenador.py
'''

import ordenador
import pytest
import contatempos

class TestaOrdenador:
  @pytest.fixture
  def o(self):
    return ordenador.Ordenador()
  
  @pytest.fixture
  def l_quase(self):
    c = contatempos.ContaTempos()
    return c.lista_quase_ordenada(1000)

  @pytest.fixture
  def l_aleatoria(self):
    c = contatempos.ContaTempos()
    return c.lista_aleatoria(1000)

  def esta_ordenada(self, l):
    for i in range(len(l)-1):
      if l[i] > l[i+1]:
        return False
      return True

  # os testes propriamente ditos
  def test_bolha_melhorada_aleatoria(self, o, l_aleatoria):
    o.bolha_melhorada(l_aleatoria)
    assert self.esta_ordenada(l_aleatoria)

  def test_selecao_direta_aleatoria(self, o, l_aleatoria):
    o.selecao_direta(l_aleatoria)
    assert self.esta_ordenada(l_aleatoria)

  def test_bolha_melhorada_quase(self, o, l_quase):
    o.bolha_melhorada(l_quase)
    assert self.esta_ordenada(l_quase)

  def test_selecao_direta_quase(self, o, l_quase):
    o.selecao_direta(l_quase)
    assert self.esta_ordenada(l_quase)
