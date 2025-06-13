#date: 2025-06-13T17:00:19Z
#url: https://api.github.com/gists/f3ba077e1d901601f70c67a8931b17b0
#owner: https://api.github.com/users/LollyTulina

import os
import random
import time


class Tabuleiro:
    def __init__(self):
        # Inicializa o tabuleiro e o tabuleiro oculto com '~' para representar a água
        self.tabuleiro = [['~' for _ in range(10)] for _ in range(10)]
        self.tabuleiro_oculto = [['~' for _ in range(10)] for _ in range(10)]
        self.navios = []
        self.memoria = []
        self.posicoes_acertadas_agua = False
        self.exibir_erros_jogador = True

        # Contadores para estatísticas
        self.tiros_recebidos = 0
        self.acertos_recebidos = 0
        self.navios_afundados = 0

    def exibir_tabuleiro(self):
        print("\033[33m  A B C D E F G H I J\033[m")
        i = 1
        for linha in self.tabuleiro:
            # Substitui '_' por '~' para exibição (bugfix)
            print(f"{i} {' '.join(linha).replace('_', '~')}")
            i += 1
        print("")

    def exibir_tabuleiro_oculto(self):
        print("\033[33m  A B C D E F G H I J\033[m")
        i = 1
        for linha in self.tabuleiro_oculto:
            # Substitui '_' por '~' para exibição (bugfix)
            print(f"{i} {' '.join(linha).replace('_', '~')}")
            i += 1
        print("")

    def posicionar_navio(self, orientacao, tamanho_navio, linha, coluna):
        partes_navio = []

        if orientacao == "h":
            for i in range(coluna, coluna + tamanho_navio):
                partes_navio.append((linha - 1, i))
                self.tabuleiro[linha - 1][i] = 'N'

        elif orientacao == "v":
            for i in range(linha, linha + tamanho_navio):
                partes_navio.append((i - 1, coluna))
                self.tabuleiro[i - 1][coluna] = 'N'

        self.navios.append({'tamanho': tamanho_navio, 'partes': partes_navio})

    def posicionar_navio_aleatorio(self, tamanho_navio):
        orientacao = random.choice(["h", "v"])
        try:
            if orientacao == "h":
                linha = random.randint(1, len(self.tabuleiro))
                coluna = random.randint(1, len(self.tabuleiro[0]) - tamanho_navio + 1)
            else:
                linha = random.randint(1, len(self.tabuleiro) - tamanho_navio + 1)
                coluna = random.randint(1, len(self.tabuleiro[0]))

            if self.verificar_posicao_disponivel(orientacao, tamanho_navio, linha, coluna):
                self.posicionar_navio(orientacao, tamanho_navio, linha, coluna)
            else:
                self.posicionar_navio_aleatorio(tamanho_navio)
        except (ValueError, IndexError):
            self.posicionar_navio_aleatorio(tamanho_navio)

    def verificar_posicao_disponivel(self, orientacao, tamanho_navio, linha, coluna):
        linha = int(linha)
        if (
                (orientacao == "h" and coluna + tamanho_navio > len(self.tabuleiro[0])) or
                (orientacao == "v" and linha + tamanho_navio - 1 > len(self.tabuleiro))
        ):
            if self.exibir_erros_jogador:
                print("\033[31mO navio não cabe no tabuleiro\033[m")
                print("")
            return False
        if orientacao == "h":
            for i in range(coluna, coluna + tamanho_navio):
                if self.tabuleiro[linha - 1][i - 1] != '~':
                    if self.exibir_erros_jogador:
                        print("\033[30mJá há uma peça aqui!\033[m")
                    return False
        elif orientacao == "v":
            for i in range(linha, linha + tamanho_navio):
                if self.tabuleiro[i - 1][coluna] != '~':
                    if self.exibir_erros_jogador:
                        print("\033[33mJá há uma peça aqui\033[m")
                    return False
        else:
            if self.exibir_erros_jogador:
                print("\033[31mA orientação não é válida!\033[m")
            return False
        return True

    def acertou_ou_errou(self, linha, coluna):
        # Verifica se a posição já foi atacada
        if self.tabuleiro[linha][coluna] == 'O' or self.tabuleiro[linha][coluna] == 'X':
            print("\033[33mVocê já tentou essa posição. Tente novamente.\033[m")
            print("")
            return False

        # Incrementa o contador de tiros recebidos (apenas para jogadas válidas)
        self.tiros_recebidos += 1

        # Verifica se acertou um navio ('N')
        if self.tabuleiro[linha][coluna] == 'N':
            print("\033[32mPARABÉNS! Você acertou um navio.\033[m")
            print("")
            self.tabuleiro_oculto[linha][coluna] = 'X'
            self.tabuleiro[linha][coluna] = 'X'

            # Incrementa o contador de acertos
            self.acertos_recebidos += 1

            self.verificar_se_afundou(linha, coluna)
            return True

        # Verifica se acertou a água ('~')
        elif self.tabuleiro[linha][coluna] == '~':
            print("\033[36mVocê acertou a água.\033[m")
            print("")
            self.tabuleiro_oculto[linha][coluna] = 'O'
            self.tabuleiro[linha][coluna] = 'O'
            return True

        return False

    def verificar_se_afundou(self, linha, coluna):
        for navio in self.navios:
            if (linha, coluna) in navio['partes']:
                navio['partes'].remove((linha, coluna))
                if not navio['partes']:
                    print(f"\033[30mVocê afundou um navio de tamanho {navio['tamanho']}!\033[m")
                    print("")
                    self.navios.remove(navio)

                    # Incrementa o contador de navios afundados
                    self.navios_afundados += 1

    def verificar_fim_de_jogo(self):
        """Verifica se não há mais navios no tabuleiro."""
        return not self.navios


class Computador:
    @staticmethod
    def jogada_computador(tabuleiro_jogador):
        if not tabuleiro_jogador.memoria:
            tabuleiro_jogador.posicoes_acertadas_agua = False

        if len(tabuleiro_jogador.memoria) == 1:
            print("\033[34mComputador\033[m: \033[30mMemória tem 1 acerto. Tentando encontrar navio...\033[m")
            linha_mem = tabuleiro_jogador.memoria[0]['linha']
            coluna_mem = tabuleiro_jogador.memoria[0]['coluna']
            Computador.tentar_encontrar_navio(tabuleiro_jogador, linha_mem, coluna_mem)
        elif len(tabuleiro_jogador.memoria) > 1:
            print("\033[34mComputador\033[m: \033[30mMemória tem >1 acerto. Verificando orientação...\033[m")
            Computador.verificar_orientacao_memoria(tabuleiro_jogador)
        else:
            print("\033[34mComputador\033[m: \033[30mMemória vazia. Jogada aleatória...\033[m")
            Computador.jogada_aleatoria(tabuleiro_jogador)

    @staticmethod
    def jogada_aleatoria(tabuleiro_jogador):
        celulas_disponiveis = []
        for r in range(len(tabuleiro_jogador.tabuleiro)):
            for c in range(len(tabuleiro_jogador.tabuleiro[0])):
                # Verifica posições não atacadas com os novos símbolos ('O', 'X')
                if tabuleiro_jogador.tabuleiro[r][c] not in ['O', 'X']:
                    celulas_disponiveis.append({'linha': r, 'coluna': c})

        if not celulas_disponiveis:
            print("Computador: Não há mais células disponíveis para atacar.")
            return

        jogada_escolhida = random.choice(celulas_disponiveis)
        linha = jogada_escolhida['linha']
        coluna = jogada_escolhida['coluna']

        print(f"\033[34mComputador\033[m: \033[30mJogada aleatória escolhida: ({linha},{coluna})\033[m")

        # Verifica se a posição contém um navio 'N' para adicionar à memória
        if tabuleiro_jogador.tabuleiro[linha][coluna] == 'N':
            print(f"\033[34mComputador\033[m: \033[30mPosição ({linha},{coluna}) contém 'N'. Adicionando à memória.\033[m")
            tabuleiro_jogador.memoria.append({'linha': linha, 'coluna': coluna})

        tabuleiro_jogador.acertou_ou_errou(linha, coluna)

    @staticmethod
    def tentar_encontrar_navio(tabuleiro_jogador, linha_base, coluna_base):
        vizinhos = [(-1, 0, "acima"), (1, 0, "abaixo"), (0, -1, "esquerda"), (0, 1, "direita")]
        random.shuffle(vizinhos)

        for dr, dc, nome_direcao in vizinhos:
            linha_alvo, coluna_alvo = linha_base + dr, coluna_base + dc

            if 0 <= linha_alvo < len(tabuleiro_jogador.tabuleiro) and \
                    0 <= coluna_alvo < len(tabuleiro_jogador.tabuleiro[0]):

                # Verifica posições não atacadas
                if tabuleiro_jogador.tabuleiro[linha_alvo][coluna_alvo] not in ['O', 'X']:
                    print(
                        f"\033[34mComputador\033[m: \033[30mTentando encontrar navio em ({linha_alvo},{coluna_alvo})\033[m")

                    # Verifica se é um navio 'N'
                    if tabuleiro_jogador.tabuleiro[linha_alvo][coluna_alvo] == 'N':
                        print(
                            f"\033[34mComputador\033[m: \033[30mPosição ({linha_alvo},{coluna_alvo}) contém 'N'. Adicionando à memória.\033[m")
                        tabuleiro_jogador.memoria.append({'linha': linha_alvo, 'coluna': coluna_alvo})

                    tabuleiro_jogador.acertou_ou_errou(linha_alvo, coluna_alvo)
                    return

        todas_adj_atacadas_agua = True
        for dr, dc, _ in vizinhos:
            la, ca = linha_base + dr, coluna_base + dc
            if 0 <= la < len(tabuleiro_jogador.tabuleiro) and 0 <= ca < len(tabuleiro_jogador.tabuleiro[0]):
                # Verifica posições não atacadas e acertos
                if tabuleiro_jogador.tabuleiro[la][ca] not in ['O', 'X']:
                    todas_adj_atacadas_agua = False
                    break
                if tabuleiro_jogador.tabuleiro[la][ca] == 'X':
                    todas_adj_atacadas_agua = False
                    break

        if todas_adj_atacadas_agua and len(tabuleiro_jogador.memoria) == 1:
            print(
                f"\033[34mComputador\033[m: \033[30mNavio em ({linha_base},{coluna_base}) parece ser de tamanho 1. Limpando memória.\033[m")
            tabuleiro_jogador.memoria = []
            tabuleiro_jogador.posicoes_acertadas_agua = False

    @staticmethod
    def encontrar_proxima_posicao_vazia(tabuleiro_jogador, linha_ref, coluna_ref, direcao):
        if not tabuleiro_jogador.memoria:
            return None

        linha_ref = int(linha_ref)
        coluna_ref = int(coluna_ref)

        coords_memoria = sorted([(int(m['linha']), int(m['coluna'])) for m in tabuleiro_jogador.memoria],
                                key=lambda p: (p[0], p[1]))
        min_linha, min_coluna = coords_memoria[0]
        max_linha, max_coluna = coords_memoria[-1]

        if direcao == 'horizontal':
            if not tabuleiro_jogador.posicoes_acertadas_agua:
                for c in range(max_coluna + 1, len(tabuleiro_jogador.tabuleiro[0])):
                    # Verifica posições não atacadas
                    if tabuleiro_jogador.tabuleiro[linha_ref][c] not in ['O', 'X']:
                        return {'linha': linha_ref, 'coluna': c}
                for c in range(min_coluna - 1, -1, -1):
                    if tabuleiro_jogador.tabuleiro[linha_ref][c] not in ['O', 'X']:
                        return {'linha': linha_ref, 'coluna': c}
            else:
                for c in range(min_coluna - 1, -1, -1):
                    if tabuleiro_jogador.tabuleiro[linha_ref][c] not in ['O', 'X']:
                        return {'linha': linha_ref, 'coluna': c}

        elif direcao == 'vertical':
            if not tabuleiro_jogador.posicoes_acertadas_agua:
                for r in range(max_linha + 1, len(tabuleiro_jogador.tabuleiro)):
                    # Verifica posições não atacadas
                    if tabuleiro_jogador.tabuleiro[r][coluna_ref] not in ['O', 'X']:
                        return {'linha': r, 'coluna': coluna_ref}
                for r in range(min_linha - 1, -1, -1):
                    if tabuleiro_jogador.tabuleiro[r][coluna_ref] not in ['O', 'X']:
                        return {'linha': r, 'coluna': coluna_ref}
            else:
                for r in range(min_linha - 1, -1, -1):
                    if tabuleiro_jogador.tabuleiro[r][coluna_ref] not in ['O', 'X']:
                        return {'linha': r, 'coluna': coluna_ref}
        return None

    @staticmethod
    def verificar_orientacao_memoria(tabuleiro_jogador):
        if len(tabuleiro_jogador.memoria) < 2:
            Computador.jogada_computador(tabuleiro_jogador)
            return

        memoria_ordenada = sorted(tabuleiro_jogador.memoria, key=lambda p: (p['linha'], p['coluna']))
        primeiro_acerto = memoria_ordenada[0]
        ultimo_acerto = memoria_ordenada[-1]
        linha1, coluna1 = primeiro_acerto['linha'], primeiro_acerto['coluna']
        linha_ult, coluna_ult = ultimo_acerto['linha'], ultimo_acerto['coluna']
        pos_alvo = None

        if linha1 == linha_ult:
            print(f"Computador: Orientação detectada: Horizontal (linha {linha1})")
            pos_alvo = Computador.encontrar_proxima_posicao_vazia(tabuleiro_jogador, linha1, coluna1, 'horizontal')
        elif coluna1 == coluna_ult:
            print(f"Computador: Orientação detectada: Vertical (coluna {coluna1})")
            pos_alvo = Computador.encontrar_proxima_posicao_vazia(tabuleiro_jogador, linha1, coluna1, 'vertical')
        else:
            print("Computador: Acertos na memória não estão alinhados. Limpando memória.")
            tabuleiro_jogador.memoria = []
            tabuleiro_jogador.posicoes_acertadas_agua = False
            Computador.jogada_aleatoria(tabuleiro_jogador)
            return

        if pos_alvo:
            linha_ataque, coluna_ataque = pos_alvo['linha'], pos_alvo['coluna']
            print(f"Computador: Atacando próxima posição em ({linha_ataque},{coluna_ataque})")

            # Verifica se era um navio 'N'
            era_navio = tabuleiro_jogador.tabuleiro[linha_ataque][coluna_ataque] == 'N'
            tabuleiro_jogador.acertou_ou_errou(linha_ataque, coluna_ataque)

            # Verifica acerto 'X'
            if tabuleiro_jogador.tabuleiro[linha_ataque][coluna_ataque] == 'X':
                if era_navio:
                    tabuleiro_jogador.memoria.append({'linha': linha_ataque, 'coluna': coluna_ataque})
                tabuleiro_jogador.posicoes_acertadas_agua = False

            # Verifica erro 'O'
            elif tabuleiro_jogador.tabuleiro[linha_ataque][coluna_ataque] == 'O':
                print(
                    f"\033[34mComputador\033[m: \033[30mAcertou água em ({linha_ataque},{coluna_ataque}) ao seguir orientação.\033[m")
                if tabuleiro_jogador.posicoes_acertadas_agua:
                    print("\033[34mComputador\033[m: \033[30mAcertou água em ambas as direções. Limpando memória.\033[m")
                    tabuleiro_jogador.memoria = []
                    tabuleiro_jogador.posicoes_acertadas_agua = False
                else:
                    tabuleiro_jogador.posicoes_acertadas_agua = True
        else:
            print(
                "\033[34mComputador\033[m: \033[30mNão encontrou próxima posição vazia. Navio pode estar cercado/afundado.\033[m")
            if tabuleiro_jogador.posicoes_acertadas_agua:
                print("\033[34mComputador\033[m: \033[30mAmbas as extremidades parecem delimitadas. Limpando memória.\033[m")
                tabuleiro_jogador.memoria = []
                tabuleiro_jogador.posicoes_acertadas_agua = False
            else:
                tabuleiro_jogador.posicoes_acertadas_agua = True


# --- Funções Auxiliares para Entrada do Usuário (com recursão) ---

def obter_orientacao_valida(tamanho_navio):
    orientacao = input(f"\033[34mDigite a orientação para o navio de tamanho {tamanho_navio} (h/v): \033[m").lower()
    if orientacao in ["h", "v"]:
        return orientacao
    else:
        print("\033[30;41mOrientação inválida. Digite 'h' para horizontal ou 'v' para vertical.\033[m")
        return obter_orientacao_valida(tamanho_navio)  # Chamada recursiva


def obter_posicao_valida():
    posicao = input("\033[33mDigite a posição da primeira parte do navio (ex: A1, J10):\033[m ").lower()
    try:
        linha = int(posicao[1:])
        coluna = ord(posicao[0].upper()) - ord('A')
        if not (0 <= coluna < 10):
            raise ValueError("\033[31mColuna inválida\033[m. Use letras de \033[33mA\033[m a \033[36mJ\033[m.")
        return linha, coluna
    except (ValueError, IndexError):
        print("\033[7;30;41mEntrada inválida. Certifique-se de seguir o formato correto.\033[m")
        return obter_posicao_valida()  # Chamada recursiva


def obter_jogada_valida():
    jogada = input("\033[33mDigite a posição para atacar (ex: A1, J10):\033[m ").lower()
    try:
        linha = int(jogada[1:]) - 1
        coluna = ord(jogada[0].upper()) - ord('A')
        if not (0 <= linha < 10 and 0 <= coluna < 10):
            raise ValueError("\033[33mPosição fora dos limites do tabuleiro.\033[m")
        return linha, coluna
    except (ValueError, IndexError):
        print("Entrada inválida. Certifique-se de seguir o formato correto.")
        return obter_jogada_valida()  # Chamada recursiva


# --- Funções Principais do Jogo (com recursão) ---

def criacao_tabuleiro_jogador(tabuleiro_jogador, navios_predefinidos):
    print("\033[34m\nJogador, posicione seus navios:\033[m")
    for tamanho_navio, quantidade in navios_predefinidos:
        for i in range(quantidade):
            print(f"\033[30;43m\nPosicionando Navio de tamanho {tamanho_navio} ({i + 1}/{quantidade})\033[m")
            tabuleiro_jogador.exibir_tabuleiro()

            # Função interna para usar recursão no posicionamento
            def tentar_posicionar_navio():
                orientacao = obter_orientacao_valida(tamanho_navio)
                linha, coluna = obter_posicao_valida()

                if tabuleiro_jogador.verificar_posicao_disponivel(orientacao, tamanho_navio, linha, coluna):
                    tabuleiro_jogador.posicionar_navio(orientacao, tamanho_navio, linha, coluna)
                else:
                    print("Posição indisponível. Tente novamente.")
                    tentar_posicionar_navio()  # Chamada recursiva

            tentar_posicionar_navio()  # Primeira chamada

    print("\nSeu tabuleiro final está pronto!")
    tabuleiro_jogador.exibir_tabuleiro()


def criacao_tabuleiro_computador(tabuleiro_computador, navios_predefinidos):
    print("\033[31mO computador está posicionando os navios...\033[m")
    tabuleiro_computador.exibir_erros_jogador = False
    for tamanho_navio, quantidade in navios_predefinidos:
        for _ in range(quantidade):
            tabuleiro_computador.posicionar_navio_aleatorio(tamanho_navio)
    print("\033[32mO computador já posicionou os navios.\033[m")
    print("")


# --- Lógica de Turno (com recursão) ---

def jogada_jogador(tabuleiro_adversario, nome_jogador):
    print("\n" + "-" * 10 + f" Vez do \033[30m{nome_jogador}\033[m " + "-" * 10)
    print("\033[34mSeu registro de ataques ao oponente:\033[")
    tabuleiro_adversario.exibir_tabuleiro_oculto()

    # Função interna para usar recursão na jogada
    def tentar_jogada():
        linha, coluna = obter_jogada_valida()
        if not tabuleiro_adversario.acertou_ou_errou(linha, coluna):
            # A mensagem de erro já é mostrada dentro de acertou_ou_errou
            tentar_jogada()  # Chamada recursiva se a jogada for inválida

    tentar_jogada()  # Primeira chamada


def exibir_estatisticas(nome_vencedor, nome_perdedor, tab_vencedor, tab_perdedor):
    print("\033[7;31;40m--- ESTATÍSTICAS DA PARTIDA ---\033[m")
    print("\n" + "=" * 15 + f" \033[32m{nome_vencedor} (VENCEDOR)\033[m " + "=" * 15)
    tiros_dados_vencedor = tab_perdedor.tiros_recebidos
    acertos_vencedor = tab_perdedor.acertos_recebidos
    navios_afundados_vencedor = tab_perdedor.navios_afundados

    print(f"\033[36mTiros realizados:\033[m {tiros_dados_vencedor}")
    print(f"\033[36mTiros certeiros:\033[m {acertos_vencedor}")
    print(f"\033[36mNavios afundados:\033[m {navios_afundados_vencedor}")

    print("\n" + "=" * 15 + f" \033[31m{nome_perdedor} (PERDEDOR)\033[m " + "=" * 15)
    tiros_dados_perdedor = tab_vencedor.tiros_recebidos
    acertos_perdedor = tab_vencedor.acertos_recebidos
    navios_afundados_perdedor = tab_vencedor.navios_afundados

    print(f"\033[36mTiros realizados:\033[m {tiros_dados_perdedor}")
    print(f"\033[36mTiros certeiros:\033[m {acertos_perdedor}")
    print(f"\033[36mNavios afundados:\033[m {navios_afundados_perdedor}")
    print("=" * 52)


# Game loop principal usando recursão
def rodada_jogo(tipo_de_jogo, tabuleiro_jogador, tabuleiro_adversario):
    # Condição de parada (caso base da recursão)
    if tabuleiro_adversario.verificar_fim_de_jogo():
        print("\n" + "#" * 20 + " \033[37mFIM DE JOGO!\033[m " + "#" * 20)
        if tipo_de_jogo == '1':
            exibir_estatisticas("Jogador", "Computador", tabuleiro_jogador, tabuleiro_adversario)
        else:
            exibir_estatisticas("Jogador 1", "Jogador 2", tabuleiro_jogador, tabuleiro_adversario)
        return  # Encerra a recursão

    if tabuleiro_jogador.verificar_fim_de_jogo():
        print("\n" + "#" * 20 + " \033[37mFIM DE JOGO!\033[m " + "#" * 20)
        if tipo_de_jogo == '1':
            exibir_estatisticas("Computador", "Jogador", tabuleiro_adversario, tabuleiro_jogador)
        else:
            exibir_estatisticas("Jogador 2", "Jogador 1", tabuleiro_adversario, tabuleiro_jogador)
        return  # Encerra a recursão

    # Jogada do jogador 1
    nome_jogador1 = "Jogador 1" if tipo_de_jogo == '2' else "Jogador"
    jogada_jogador(tabuleiro_adversario, nome_jogador1)

    # Verifica novamente se o jogador 1 venceu antes de passar o turno
    if tabuleiro_adversario.verificar_fim_de_jogo():
        rodada_jogo(tipo_de_jogo, tabuleiro_jogador,
                    tabuleiro_adversario)  # Chamada recursiva para mostrar placar final
        return

    # Jogada do jogador 2 (ou computador)
    if tipo_de_jogo == '1':
        print("\n\033[30mVez do Computador...\033[m")
        time.sleep(2)
        Computador.jogada_computador(tabuleiro_jogador)
        print("\n\033[30mSeu tabuleiro após o ataque inimigo:\033[m")
        tabuleiro_jogador.exibir_tabuleiro()
    elif tipo_de_jogo == '2':
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\033[34mPasse o controle para o Jogador 2.\033[m")
        input("\033[32mPressione Enter para continuar...\033[m")
        nome_jogador2 = "Jogador 2"
        jogada_jogador(tabuleiro_jogador, nome_jogador2)
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\033[34mPasse o controle de volta para o Jogador 1.\033[m")
        input("\033[32mPressione Enter para continuar...\033[m")

    # Chamada recursiva para a próxima rodada
    rodada_jogo(tipo_de_jogo, tabuleiro_jogador, tabuleiro_adversario)


# --- Função de Inicialização ---

def play_battleship():
    print("\033[0;34;33mBem-vindo ao jogo Batalha Naval!\033[m")

    # Função interna para escolher o modo de jogo com recursão
    def escolher_modo():
        tipo_de_jogo = input(
            "\033[36mComo deseja jogar?\033[m \033[32mUm Jogador (1)\033[m | \033[31mDois Jogadores (2)\033[m: ")
        if tipo_de_jogo in ['1', '2']:
            return tipo_de_jogo
        else:
            print("\033[30;41mOpção inválida. Por favor, escolha '1' ou '2'.\033[m")
            return escolher_modo()

    tipo_de_jogo = escolher_modo()

    navios_predefinidos = [(5, 1), (4, 1), (3, 2), (2, 2)]

    if tipo_de_jogo == '1':
        tabuleiro_jogador = Tabuleiro()
        tabuleiro_computador = Tabuleiro()
        criacao_tabuleiro_jogador(tabuleiro_jogador, navios_predefinidos)
        criacao_tabuleiro_computador(tabuleiro_computador, navios_predefinidos)
        input("\033[30;42mTudo pronto! Pressione Enter para começar a partida...\033[m")
        rodada_jogo(tipo_de_jogo, tabuleiro_jogador, tabuleiro_computador)
    elif tipo_de_jogo == '2':
        tabuleiro_jogador1 = Tabuleiro()
        tabuleiro_jogador2 = Tabuleiro()

        criacao_tabuleiro_jogador(tabuleiro_jogador1, navios_predefinidos)
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\033[30mAgora é a vez do Jogador 2.\033[m")
        input("\033[32mPressione Enter para o Jogador 2 posicionar seus navios...\033[m")
        criacao_tabuleiro_jogador(tabuleiro_jogador2, navios_predefinidos)
        os.system('cls' if os.name == 'nt' else 'clear')

        input("\033[32;40mTudo pronto! Pressione Enter para começar a partida...\033[m")
        rodada_jogo(tipo_de_jogo, tabuleiro_jogador1, tabuleiro_jogador2)


# Executar o jogo
play_battleship()