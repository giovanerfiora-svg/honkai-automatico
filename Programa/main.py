"""
Simulador de Times - Honkai: Star Rail (ou jogo similar)

Este programa:
  1) Treina um modelo (RandomForestRegressor) com base no histórico de resultados.
  2) Mostra as melhores combinações de time SIMULADAS pelo modelo (ainda não testadas).
  3) Permite TESTAR um time específico e ver a pontuação prevista pelo modelo.
  4) Permite CADASTRAR uma nova tentativa real (3 resultados) e atualiza o CSV.

Como usar:
    python simulador_times.py

O arquivo de dados (resultados.csv) deve estar no mesmo formato usado em modelo.py:
    id_registro,batalha,efeito_buff,personagem_1,personagem_2,personagem_3,personagem_4,
    tentativa_1,tentativa_2,tentativa_3,pontuacao_media,pontuacao_maior,desvio_padrao
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

# Ajuste este caminho se o seu CSV estiver em outra pasta (ex: "Tabelas/resultados.csv")
ARQUIVO_CSV = "Tabelas/resultados.csv"

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
# Treinamento do modelo
# --------------------------------------------------------------------------- #
def carregar_dados():
    if not os.path.exists(ARQUIVO_CSV):
        raise FileNotFoundError(
            f"Não encontrei o arquivo '{ARQUIVO_CSV}'. "
            "Ajuste a variável ARQUIVO_CSV no topo do script."
        )
    return pd.read_csv(ARQUIVO_CSV)


def treinar_modelo(df):
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

    pipeline.fit(X, y)
    return pipeline


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def escolher_da_lista(opcoes, titulo, permitir_multiplo=False):
    """Mostra uma lista numerada e devolve a(s) escolha(s) do usuário."""
    opcoes = list(opcoes)
    print(f"\n{titulo}")
    for i, opcao in enumerate(opcoes, start=1):
        print(f"  [{i}] {opcao}")

    if not permitir_multiplo:
        while True:
            escolha = input("Escolha o número: ").strip()
            if escolha.isdigit() and 1 <= int(escolha) <= len(opcoes):
                return opcoes[int(escolha) - 1]
            print("Opção inválida, tente novamente.")
    else:
        while True:
            escolha = input(
                "Escolha os números separados por vírgula (ex: 1,3,5,7): "
            ).strip()
            indices = [x.strip() for x in escolha.split(",") if x.strip()]
            if all(x.isdigit() and 1 <= int(x) <= len(opcoes) for x in indices):
                return [opcoes[int(x) - 1] for x in indices]
            print("Entrada inválida, tente novamente.")


def gerar_id_registro():
    return f"e_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


# --------------------------------------------------------------------------- #
# Opção 1: Melhores resultados simulados
# --------------------------------------------------------------------------- #
def melhores_resultados_simulados(df, pipeline):
    batalhas = sorted(df["batalha"].unique())
    batalha = escolher_da_lista(batalhas, "Para qual batalha você quer simular?")

    df_batalha = df[df["batalha"] == batalha]
    buffs = sorted(df_batalha["efeito_buff"].unique())
    personagens = sorted(
        pd.unique(
            df_batalha[
                ["personagem_1", "personagem_2", "personagem_3", "personagem_4"]
            ].values.ravel()
        )
    )

    if len(personagens) < 4:
        print("Não há personagens suficientes no histórico dessa batalha para simular.")
        return

    try:
        top_n = int(input("Quantas melhores combinações mostrar? (padrão 10): ") or 10)
    except ValueError:
        top_n = 10

    # Combinações já testadas, para sinalizar quais são realmente inéditas
    ja_testados = set(
        tuple(sorted(row))
        for row in df_batalha[
            ["personagem_1", "personagem_2", "personagem_3", "personagem_4"]
        ].values
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
    df_sim["pontuacao_prevista"] = pipeline.predict(X_sim)

    df_sim = df_sim.sort_values("pontuacao_prevista", ascending=False).head(top_n)

    print(f"\n=== Top {top_n} combinações simuladas para {batalha} ===")
    for _, linha in df_sim.iterrows():
        status = "(já testado)" if linha["ja_testado"] else "(inédito)"
        print(
            f"  {linha['pontuacao_prevista']:.1f} pts | {linha['efeito_buff']} | "
            f"{linha['personagem_1']}, {linha['personagem_2']}, "
            f"{linha['personagem_3']}, {linha['personagem_4']} {status}"
        )


# --------------------------------------------------------------------------- #
# Opção 2: Testar times específicos
# --------------------------------------------------------------------------- #
def testar_time(df, pipeline):
    batalhas = sorted(df["batalha"].unique())
    batalha = escolher_da_lista(batalhas, "Qual batalha?")

    df_batalha = df[df["batalha"] == batalha]
    buffs = sorted(df_batalha["efeito_buff"].unique())
    buff = escolher_da_lista(buffs, "Qual efeito de buff?")

    personagens = sorted(
        pd.unique(
            df[["personagem_1", "personagem_2", "personagem_3", "personagem_4"]]
            .values.ravel()
        )
    )

    print(
        "\nAgora escolha os 4 personagens do time (ou digite um nome novo, "
        "caso ele ainda não exista na lista)."
    )
    time_escolhido = []
    for i in range(1, 5):
        print(f"\nPersonagem {i}:")
        for idx, p in enumerate(personagens, start=1):
            print(f"  [{idx}] {p}")
        escolha = input(
            "Digite o número da lista OU digite o nome de um personagem novo: "
        ).strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(personagens):
            time_escolhido.append(personagens[int(escolha) - 1])
        else:
            time_escolhido.append(escolha)

    entrada = pd.DataFrame(
        [
            {
                "batalha": batalha,
                "efeito_buff": buff,
                "personagem_1": time_escolhido[0],
                "personagem_2": time_escolhido[1],
                "personagem_3": time_escolhido[2],
                "personagem_4": time_escolhido[3],
            }
        ]
    )

    previsao = pipeline.predict(entrada[COLUNAS_CATEGORICAS])[0]
    print(f"\n>>> Pontuação prevista pelo modelo: {previsao:.1f} pts")


# --------------------------------------------------------------------------- #
# Opção 3: Cadastrar tentativa real
# --------------------------------------------------------------------------- #
def cadastrar_tentativa(df):
    batalhas = sorted(df["batalha"].unique())
    print("\nBatalhas existentes:", ", ".join(batalhas) if batalhas else "(nenhuma ainda)")
    batalha = input("Nome da batalha (ex: '1ª Batalha'): ").strip()

    buffs = sorted(df["efeito_buff"].unique())
    print("Efeitos de buff existentes:", ", ".join(buffs) if buffs else "(nenhum ainda)")
    efeito_buff = input("Nome do efeito de buff: ").strip()

    personagens = []
    for i in range(1, 5):
        personagens.append(input(f"Personagem {i}: ").strip())

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
    media = pontuacao_media
    desvio_padrao = round(
        (sum((t - media) ** 2 for t in tentativas) / 3) ** 0.5, 2
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
    print("Treinando modelo com os dados existentes...")
    pipeline = treinar_modelo(df)
    print(f"Modelo treinado com {len(df)} registros.\n")

    while True:
        print("\n===================== MENU =====================")
        print("[1] Melhores resultados simulados")
        print("[2] Testar tais times")
        print("[3] Cadastre sua tentativa aqui")
        print("[4] Sair")
        print("==================================================")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            melhores_resultados_simulados(df, pipeline)
        elif opcao == "2":
            testar_time(df, pipeline)
        elif opcao == "3":
            df = cadastrar_tentativa(df)
            print("Retreinando modelo com o novo registro...")
            pipeline = treinar_modelo(df)
        elif opcao == "4":
            print("Até mais!")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()