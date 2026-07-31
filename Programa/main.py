"""
Simulador de Times - Honkai: Star Rail (ou jogo similar)

Este programa:
  1) Treina um modelo (RandomForestRegressor) com base no histórico de resultados.
  2) Mostra as melhores combinações de time SIMULADAS pelo modelo (ainda não testadas).
  3) Permite CADASTRAR uma nova tentativa real (3 resultados) e atualiza o CSV.

O modelo é RETREINADO a cada novo cadastro (veja o main()), então ele "aprende"
ao longo do tempo: quanto mais tentativas reais forem registradas, mais dados
o RandomForest tem para refinar suas previsões de pontuação.

As opções de batalha, efeito de buff e personagens vêm das tabelas de referência
(batalhas.csv, efeitos.csv, personagens.csv), não apenas do histórico — assim dá
para simular/cadastrar combinações que ainda não têm nenhum resultado registrado.

Como usar:
    python simulador_times.py

Arquivos esperados na mesma pasta (ajuste os caminhos abaixo se necessário):
    resultados.csv   -> id_registro,batalha,efeito_buff,personagem_1..4,
                         tentativa_1..3,pontuacao_media,pontuacao_maior,desvio_padrao
    batalhas.csv      -> id_batalha,nome_batalha
    efeitos.csv       -> id_efeito,id_batalha,nome_efeito
    personagens.csv   -> id,nome
"""

import itertools
import os
import random
import time

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Ajuste estes caminhos se os arquivos estiverem em outra pasta (ex: "Tabelas/resultados.csv")
ARQUIVO_CSV = "Tabelas/resultados.csv"
ARQUIVO_BATALHAS = "Tabelas/batalhas.csv"
ARQUIVO_EFEITOS = "Tabelas/efeitos.csv"
ARQUIVO_PERSONAGENS = "Tabelas/personagens.csv"

COLUNAS_CATEGORICAS = [
    "batalha",
    "efeito_buff",
    "personagem_1",
    "personagem_2",
    "personagem_3",
    "personagem_4",
]

COLUNAS_TENTATIVAS = ["tentativa_1", "tentativa_2", "tentativa_3"]


# --------------------------------------------------------------------------- #
# Carregamento de dados e tabelas de referência
# --------------------------------------------------------------------------- #
def _ler_csv_obrigatorio(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Não encontrei o arquivo '{caminho}'. "
            "Ajuste o caminho correspondente no topo do script."
        )
    return pd.read_csv(caminho)


def carregar_dados():
    return _ler_csv_obrigatorio(ARQUIVO_CSV)


def carregar_referencias():
    """Carrega as tabelas de batalhas, efeitos e personagens."""
    df_batalhas = _ler_csv_obrigatorio(ARQUIVO_BATALHAS)
    df_efeitos = _ler_csv_obrigatorio(ARQUIVO_EFEITOS)
    df_personagens = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    return df_batalhas, df_efeitos, df_personagens


def nomes_personagens(df_personagens):
    return sorted(df_personagens["nome"].tolist())


def nomes_batalhas(df_batalhas):
    return df_batalhas.sort_values("id_batalha")["nome_batalha"].tolist()


def efeitos_da_batalha(df_batalhas, df_efeitos, nome_batalha):
    """Retorna os nomes de efeito de buff cadastrados para a batalha escolhida."""
    id_batalha = df_batalhas.loc[
        df_batalhas["nome_batalha"] == nome_batalha, "id_batalha"
    ].iloc[0]
    return df_efeitos.loc[
        df_efeitos["id_batalha"] == id_batalha, "nome_efeito"
    ].tolist()


# --------------------------------------------------------------------------- #
# Treinamento do modelo
# --------------------------------------------------------------------------- #
def treinar_modelo(df):
    """
    Treina o modelo que aprende a prever a pontuação média de um time
    a partir da batalha, do efeito de buff e dos 4 personagens escolhidos.

    Como funciona, passo a passo:

    1) ENTRADAS (X) e SAÍDA (y)
       - X: as colunas categóricas que descrevem a "receita" do time
         (batalha, efeito_buff, personagem_1..4).
       - y: o valor que queremos prever, aqui a pontuação_media obtida
         nas tentativas reais já cadastradas.
       Ou seja, o modelo aprende o padrão "essa combinação de batalha +
       buff + personagens tende a gerar essa pontuação".

    2) PRÉ-PROCESSAMENTO (OneHotEncoder)
       - Os dados de entrada são todos texto (nomes de batalha, buff,
         personagens), e modelos de machine learning trabalham com números.
       - O OneHotEncoder transforma cada valor categórico em uma coluna
         binária (0 ou 1). Por exemplo, a coluna "batalha" vira várias
         colunas do tipo "batalha_X", "batalha_Y", cada uma marcando
         se aquele time é ou não daquela batalha.
       - handle_unknown="ignore" evita erro caso apareça, numa previsão
         futura, um valor que o modelo nunca viu no treino (ele
         simplesmente trata como "nenhuma das categorias conhecidas").

    3) MODELO (RandomForestRegressor)
       - É um conjunto ("floresta") de várias árvores de decisão
         (n_estimators=300 = 300 árvores).
       - Cada árvore aprende, de forma um pouco diferente (por causa de
         aleatoriedade nos dados/atributos usados), a relação entre a
         combinação de time e a pontuação.
       - A previsão final é a MÉDIA das previsões de todas as árvores,
         o que deixa o resultado mais estável e menos sujeito a "decorar"
         demais os dados de treino (overfitting) do que uma única árvore.
       - random_state=42 apenas fixa a semente aleatória, para que o
         treinamento seja reproduzível (mesmos dados -> mesmo resultado).

    4) PIPELINE
       - Encadeia o pré-processamento (OneHotEncoder) e o modelo
         (RandomForestRegressor) em um único objeto. Assim, sempre que
         formos prever algo novo, os dados passam automaticamente pelas
         mesmas transformações usadas no treino, sem precisar repetir
         esse código toda vez.

    5) APRENDIZADO CONTÍNUO
       - Esta função é chamada de novo (retreinando o modelo do zero)
         toda vez que um novo registro real é cadastrado (veja main()).
       - Como o RandomForest é treinado com TODO o histórico disponível
         a cada chamada, o modelo tende a ficar mais preciso conforme
         mais tentativas reais vão sendo cadastradas ao longo do tempo.
    """
    X = df[COLUNAS_CATEGORICAS]
    y = df["pontuacao_media"]

    preprocessador = ColumnTransformer(
        transformers=[
            ("categorias", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)
        ]
    )

    modelo = RandomForestRegressor(n_estimators=300, random_state=42)

    pipeline = Pipeline(
        [
            ("preprocessador", preprocessador),
            ("modelo", modelo),
        ]
    )

    # fit() é o passo de treinamento em si: o modelo "olha" para todas as
    # combinações já cadastradas (X) e suas pontuações reais (y) e ajusta
    # seus parâmetros internos para aprender esse padrão.
    pipeline.fit(X, y)
    return pipeline


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def escolher_da_lista(opcoes, titulo):
    """Mostra uma lista numerada e devolve a escolha do usuário."""
    opcoes = list(opcoes)
    print(f"\n{titulo}")
    for i, opcao in enumerate(opcoes, start=1):
        print(f"  [{i}] {opcao}")

    while True:
        escolha = input("Escolha o número: ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(opcoes):
            return opcoes[int(escolha) - 1]
        print("Opção inválida, tente novamente.")


def gerar_id_registro():
    return f"e_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


# --------------------------------------------------------------------------- #
# Opção 1: Melhores resultados simulados
# --------------------------------------------------------------------------- #
def melhores_resultados_simulados(df, pipeline, df_batalhas, df_efeitos, df_personagens):
    batalha = escolher_da_lista(
        nomes_batalhas(df_batalhas), "Para qual batalha você quer simular?"
    )
    buffs = efeitos_da_batalha(df_batalhas, df_efeitos, batalha)
    if not buffs:
        print("Essa batalha não tem efeitos de buff cadastrados em efeitos.csv.")
        return

    personagens = nomes_personagens(df_personagens)

    try:
        top_n = int(input("Quantas melhores combinações mostrar? (padrão 10): ") or 10)
    except ValueError:
        top_n = 10

    # Combinações já testadas de verdade nessa batalha, para sinalizar o que é inédito
    df_batalha_hist = df[df["batalha"] == batalha]
    ja_testados = set(
        tuple(sorted(row))
        for row in df_batalha_hist[
            ["personagem_1", "personagem_2", "personagem_3", "personagem_4"]
        ].values
    )

    print(
        f"\nGerando combinações ({len(personagens)} personagens x {len(buffs)} "
        f"efeitos)... isso pode levar alguns segundos."
    )

    linhas_simuladas = []
    for buff in buffs:
        for combo in itertools.combinations(personagens, 4):
            linhas_simuladas.append(
                {
                    "batalha": batalha,
                    "efeito_buff": buff,
                    "personagem_1": combo[0],
                    "personagem_2": combo[1],
                    "personagem_3": combo[2],
                    "personagem_4": combo[3],
                    "ja_testado": tuple(sorted(combo)) in ja_testados,
                }
            )

    df_sim = pd.DataFrame(linhas_simuladas)
    X_sim = df_sim[COLUNAS_CATEGORICAS]
    # predict() usa o modelo já treinado para estimar a pontuação de cada
    # combinação simulada, sem precisar jogar de verdade: os dados passam
    # pelo mesmo OneHotEncoder do treino e depois pelas 300 árvores do
    # RandomForest, que devolvem a média das previsões de cada uma.
    df_sim["pontuacao_prevista"] = pipeline.predict(X_sim)

    df_sim = df_sim.sort_values("pontuacao_prevista", ascending=False).head(top_n)

    if df_batalha_hist.empty:
        print(
            "\nAviso: ainda não há nenhum resultado real cadastrado para essa "
            "batalha, então a previsão é baseada só nos efeitos/personagens "
            "(tende a ser menos confiável)."
        )

    print(f"\n=== Top {top_n} combinações simuladas para {batalha} ===")
    for _, linha in df_sim.iterrows():
        status = "(já testado)" if linha["ja_testado"] else "(inédito)"
        print(
            f"  {linha['pontuacao_prevista']:.1f} pts | {linha['efeito_buff']} | "
            f"{linha['personagem_1']}, {linha['personagem_2']}, "
            f"{linha['personagem_3']}, {linha['personagem_4']} {status}"
        )


# --------------------------------------------------------------------------- #
# Opção 2: Cadastrar tentativa real
# --------------------------------------------------------------------------- #
def cadastrar_tentativa(df, df_batalhas, df_efeitos, df_personagens):
    batalha = escolher_da_lista(nomes_batalhas(df_batalhas), "Qual batalha?")
    buffs = efeitos_da_batalha(df_batalhas, df_efeitos, batalha)
    if not buffs:
        print("Essa batalha não tem efeitos de buff cadastrados em efeitos.csv.")
        return df
    efeito_buff = escolher_da_lista(buffs, "Qual efeito de buff?")

    personagens_disponiveis = nomes_personagens(df_personagens)
    print("\nEscolha os 4 personagens do time:")
    personagens = []
    for i in range(1, 5):
        p = escolher_da_lista(personagens_disponiveis, f"Personagem {i}:")
        personagens.append(p)

    tentativas = []
    for i in range(1, 4):
        while True:
            valor = input(f"Tentativa {i} (pontuação): ").strip()
            try:
                tentativas.append(float(valor))
                break
            except ValueError:
                print("Digite um número válido.")

    pontuacao_media = round(sum(tentativas) / 3, 2)
    pontuacao_maior = max(tentativas)
    desvio_padrao = round(
        (sum((t - pontuacao_media) ** 2 for t in tentativas) / 3) ** 0.5, 2
    )

    nova_linha = {
        "id_registro": gerar_id_registro(),
        "batalha": batalha,
        "efeito_buff": efeito_buff,
        "personagem_1": personagens[0],
        "personagem_2": personagens[1],
        "personagem_3": personagens[2],
        "personagem_4": personagens[3],
        "tentativa_1": tentativas[0],
        "tentativa_2": tentativas[1],
        "tentativa_3": tentativas[2],
        "pontuacao_media": pontuacao_media,
        "pontuacao_maior": pontuacao_maior,
        "desvio_padrao": desvio_padrao,
    }

    print("\nResumo do novo registro:")
    for chave, valor in nova_linha.items():
        print(f"  {chave}: {valor}")

    confirmar = input("\nConfirmar cadastro? (s/n): ").strip().lower()
    if confirmar != "s":
        print("Cadastro cancelado.")
        return df

    df_atualizado = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    df_atualizado.to_csv(ARQUIVO_CSV, index=False)
    print(f"Registro salvo em '{ARQUIVO_CSV}' com sucesso!")
    return df_atualizado


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main():
    df = carregar_dados()
    df_batalhas, df_efeitos, df_personagens = carregar_referencias()

    print("Treinando modelo com os dados existentes...")
    pipeline = treinar_modelo(df)
    print(f"Modelo treinado com {len(df)} registros.\n")

    while True:
        print("\n===================== MENU =====================")
        print("[1] Melhores resultados simulados")
        print("[2] Cadastre sua tentativa aqui")
        print("[3] Sair")
        print("==================================================")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            melhores_resultados_simulados(
                df, pipeline, df_batalhas, df_efeitos, df_personagens
            )
        elif opcao == "2":
            df = cadastrar_tentativa(df, df_batalhas, df_efeitos, df_personagens)
            # A cada novo registro real cadastrado, o modelo é retreinado
            # do zero com o histórico atualizado (incluindo o novo dado).
            # É assim que o programa "aprende" com o tempo: mais tentativas
            # cadastradas -> mais exemplos para o RandomForest -> previsões
            # tendem a ficar mais precisas.
            print("Retreinando modelo com o novo registro...")
            pipeline = treinar_modelo(df)
        elif opcao == "3":
            print("Até mais!")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()