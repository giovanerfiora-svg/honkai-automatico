"""
Simulador de Times Multi-Modo - Honkai: Star Rail
Suporta:
  1) Sombra Apocalíptica (Maior pontuação = Melhor)
  2) Pura Ficção (Maior pontuação = Melhor, Buffs globais)
  3) Memória do Caos (Menor número de ciclos = Melhor, Sem buffs)
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

# Caminho base das tabelas
PASTA_TABELAS = "Tabelas"
ARQUIVO_PERSONAGENS = os.path.join(PASTA_TABELAS, "personagens.csv")

# Personagens de sustentação/sustentabilidade (Curandeiros e Shielders)
CURANDEIROS = {
    "Gallagher", "Huohuo", "Hyacine", "Dan Heng (Preservação)",
    "Aventurine", "Luocha", "Fu Xuan", "Gepard", "Bailu", "Lynx", "March 7th"
}

MODOS_CONFIG = {
    "1": {
        "nome": "Sombra Apocalíptica",
        "pasta": os.path.join(PASTA_TABELAS, "sombra_apocaliptica"),
        "tem_efeitos": True,
        "efeitos_globais": False,
        "métrica_menor_é_melhor": False,
        "label_métrica": "pontuação",
        "coluna_alvo": "pontuacao_media",
        "colunas_cat": ["batalha", "efeito_buff", "personagem_1", "personagem_2", "personagem_3", "personagem_4"]
    },
    "2": {
        "nome": "Pura Ficção",
        "pasta": os.path.join(PASTA_TABELAS, "pura_ficcao"),
        "tem_efeitos": True,
        "efeitos_globais": True,
        "métrica_menor_é_melhor": False,
        "label_métrica": "pontuação",
        "coluna_alvo": "pontuacao_media",
        "colunas_cat": ["batalha", "efeito_buff", "personagem_1", "personagem_2", "personagem_3", "personagem_4"]
    },
    "3": {
        "nome": "Memória do Caos",
        "pasta": os.path.join(PASTA_TABELAS, "memoria_do_caos"),
        "tem_efeitos": False,
        "efeitos_globais": False,
        "métrica_menor_é_melhor": True,
        "label_métrica": "ciclos",
        "coluna_alvo": "ciclos_medios",
        "colunas_cat": ["batalha", "personagem_1", "personagem_2", "personagem_3", "personagem_4"]
    }
}


def time_tem_curandeiro(personagens):
    return any(p in CURANDEIROS for p in personagens)


def _ler_csv_obrigatorio(caminho):
    if not os.path.exists(caminho):
        # Se o arquivo não existir, criamos um DataFrame vazio de apoio
        return pd.DataFrame()
    return pd.read_csv(caminho)


def carregar_referencias(config):
    df_personagens = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    
    arq_efeitos = os.path.join(config["pasta"], "efeitos.csv")
    df_efeitos = _ler_csv_obrigatorio(arq_efeitos) if config["tem_efeitos"] else pd.DataFrame()
    
    arq_resultados = os.path.join(config["pasta"], "resultados.csv")
    df_resultados = _ler_csv_obrigatorio(arq_resultados)
    
    return df_personagens, df_efeitos, df_resultados


def nomes_personagens(df_personagens):
    if df_personagens.empty:
        return []
    return sorted(df_personagens["nome"].tolist())


def obter_efeitos(df_efeitos, batalha, config):
    if not config["tem_efeitos"] or df_efeitos.empty:
        return []
    if config["efeitos_globais"]:
        return df_efeitos["nome_efeito"].tolist()
    else:
        # Assumindo id_batalha 1 para 1ª Batalha e 2 para 2ª Batalha
        id_batalha = 1 if "1ª" in batalha or "1" in batalha else 2
        if "id_batalha" in df_efeitos.columns:
            return df_efeitos.loc[df_efeitos["id_batalha"] == id_batalha, "nome_efeito"].tolist()
        return df_efeitos["nome_efeito"].tolist()


def treinar_modelo(df, config):
    if df.empty or len(df) < 2:
        return None

    X = df[config["colunas_cat"]]
    y = df[config["coluna_alvo"]]

    preprocessador = ColumnTransformer(
        transformers=[
            ("categorias", OneHotEncoder(handle_unknown="ignore"), config["colunas_cat"])
        ]
    )

    modelo = RandomForestRegressor(n_estimators=300, random_state=42)

    pipeline = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", modelo)
    ])

    pipeline.fit(X, y)
    return pipeline


def escolher_da_lista(opcoes, titulo):
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


def melhores_resultados_simulados(df_resultados, pipeline, df_efeitos, df_personagens, config):
    batalhas = ["1ª Batalha", "2ª Batalha"]
    batalha = escolher_da_lista(batalhas, "Para qual batalha/lado você quer simular?")

    buffs = []
    if config["tem_efeitos"]:
        buffs = obter_efeitos(df_efeitos, batalha, config)
        if not buffs:
            print("Nenhum efeito de buff cadastrado.")
            return

    personagens = nomes_personagens(df_personagens)

    try:
        top_n = int(input("Quantas melhores combinações mostrar? (padrão 10): ") or 10)
    except ValueError:
        top_n = 10

    # Histórico de já testados
    ja_testados = set()
    if not df_resultados.empty and "batalha" in df_resultados.columns:
        df_batalha_hist = df_resultados[df_resultados["batalha"] == batalha]
        ja_testados = set(
            tuple(sorted(row))
            for row in df_batalha_hist[["personagem_1", "personagem_2", "personagem_3", "personagem_4"]].values
        )

    print("\nGerando combinações inéditas e prevendo com IA...")

    linhas_simuladas = []
    loop_buffs = buffs if config["tem_efeitos"] else [None]

    for buff in loop_buffs:
        for combo in itertools.combinations(personagens, 4):
            # Ignora times sem curandeiro/sustentação
            if not time_tem_curandeiro(combo):
                continue
            
            # FILTRO NOVO: Ignora combinações que já foram testadas anteriormente
            combo_ordenado = tuple(sorted(combo))
            if combo_ordenado in ja_testados:
                continue
            
            dado = {
                "batalha": batalha,
                "personagem_1": combo[0],
                "personagem_2": combo[1],
                "personagem_3": combo[2],
                "personagem_4": combo[3],
            }
            if config["tem_efeitos"]:
                dado["efeito_buff"] = buff
            
            linhas_simuladas.append(dado)

    if not linhas_simuladas:
        print("\n[Aviso] Não há novas combinações inéditas disponíveis com os personagens atuais!")
        return

    df_sim = pd.DataFrame(linhas_simuladas)

    if pipeline is None:
        print("\n[Aviso] Dados insuficientes para treinar a IA. Cadastre alguns resultados reais primeiro!")
        return

    df_sim["previsto"] = pipeline.predict(df_sim[config["colunas_cat"]])

    # Ordena as estimativas e seleciona apenas o Top N inédito
    ascending = config["métrica_menor_é_melhor"]
    df_sim = df_sim.sort_values("previsto", ascending=ascending).head(top_n)

    print(f"\n=== Top {len(df_sim)} combinações inéditas simuladas ({config['nome']}) ===")
    for _, linha in df_sim.iterrows():
        buff_str = f" | Buff: {linha['efeito_buff']}" if config["tem_efeitos"] else ""
        unit = "ciclos" if config["métrica_menor_é_melhor"] else "pts"
        print(
            f"  Previsão: {linha['previsto']:.2f} {unit}{buff_str} | "
            f"{linha['personagem_1']}, {linha['personagem_2']}, "
            f"{linha['personagem_3']}, {linha['personagem_4']} "
        )


def cadastrar_tentativa(df_resultados, df_efeitos, df_personagens, config):
    batalhas = ["1ª Batalha", "2ª Batalha"]
    batalha = escolher_da_lista(batalhas, "Qual batalha/lado?")

    efeito_buff = None
    if config["tem_efeitos"]:
        buffs = obter_efeitos(df_efeitos, batalha, config)
        if buffs:
            efeito_buff = escolher_da_lista(buffs, "Qual efeito de buff?")

    personagens_disponiveis = nomes_personagens(df_personagens)

    while True:
        print("\nEscolha os 4 personagens do time:")
        personagens = []
        for i in range(1, 5):
            p = escolher_da_lista(personagens_disponiveis, f"Personagem {i}:")
            personagens.append(p)

        if time_tem_curandeiro(personagens):
            break

        print("\nEsse time precisa de pelo menos um personagem de sustentação/cura!")

    tentativas = []
    print(f"\nInforme os 3 resultados ({config['label_métrica']}):")
    for i in range(1, 4):
        while True:
            valor = input(f"Tentativa {i}: ").strip()
            try:
                tentativas.append(float(valor))
                break
            except ValueError:
                print("Digite um número válido.")

    media = round(sum(tentativas) / 3, 2)
    melhor_val = min(tentativas) if config["métrica_menor_é_melhor"] else max(tentativas)
    desvio = round((sum((t - media) ** 2 for t in tentativas) / 3) ** 0.5, 2)

    nova_linha = {
        "id_registro": gerar_id_registro(),
        "batalha": batalha,
        "personagem_1": personagens[0],
        "personagem_2": personagens[1],
        "personagem_3": personagens[2],
        "personagem_4": personagens[3],
        "tentativa_1": tentativas[0],
        "tentativa_2": tentativas[1],
        "tentativa_3": tentativas[2],
        config["coluna_alvo"]: media,
        "métrica_destaque": melhor_val,
        "desvio_padrao": desvio,
    }

    if config["tem_efeitos"]:
        nova_linha["efeito_buff"] = efeito_buff

    print("\nResumo do novo registro:")
    for k, v in nova_linha.items():
        print(f"  {k}: {v}")

    if input("\nConfirmar cadastro? (s/n): ").strip().lower() != "s":
        print("Cancelado.")
        return df_resultados

    df_atualizado = pd.concat([df_resultados, pd.DataFrame([nova_linha])], ignore_index=True)
    
    os.makedirs(config["pasta"], exist_ok=True)
    arq_out = os.path.join(config["pasta"], "resultados.csv")
    df_atualizado.to_csv(arq_out, index=False)
    print(f"Registro salvo com sucesso em '{arq_out}'!")
    return df_atualizado


def menu_modo(config):
    df_personagens, df_efeitos, df_resultados = carregar_referencias(config)

    print(f"\nTreinando modelo para: {config['nome']}...")
    pipeline = treinar_modelo(df_resultados, config)

    while True:
        print(f"\n=== MODO DE JOGO: {config['nome'].upper()} ===")
        print("[1] Melhores resultados simulados (IA)")
        print("[2] Cadastrar tentativa real")
        print("[3] Voltar ao menu principal")

        op = input("Escolha uma opção: ").strip()

        if op == "1":
            melhores_resultados_simulados(df_resultados, pipeline, df_efeitos, df_personagens, config)
        elif op == "2":
            df_resultados = cadastrar_tentativa(df_resultados, df_efeitos, df_personagens, config)
            pipeline = treinar_modelo(df_resultados, config)
        elif op == "3":
            break


def main():
    while True:
        print("\n================ SELEÇÃO DE MODO ================")
        print("[1] Sombra Apocalíptica")
        print("[2] Pura Ficção")
        print("[3] Memória do Caos")
        print("[4] Sair")
        print("==================================================")

        escolha = input("Escolha o modo de jogo: ").strip()

        if escolha in MODOS_CONFIG:
            menu_modo(MODOS_CONFIG[escolha])
        elif escolha == "4":
            print("Até logo, Trailblazer!")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()