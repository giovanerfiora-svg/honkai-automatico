import os
import random
import time
import itertools
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Define o caminho base como sendo a pasta pai da pasta 'Backend'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Aponta para a pasta 'frontend' onde está o arquivo index.html
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Atualize também o caminho das tabelas para apontar para a pasta "Dados":
PASTA_TABELAS = os.path.join(BASE_DIR, "Dados")
ARQUIVO_PERSONAGENS = os.path.join(PASTA_TABELAS, "personagens.csv")

CURANDEIROS = {
    "Gallagher", "Huohuo", "Hyacine", "Dan Heng (Preservação)",
    "Aventurine", "Luocha", "Fu Xuan", "Gepard", "Bailu", "Lynx", "March 7th"
}

MODOS_CONFIG = {
    "sombra_apocaliptica": {
        "nome": "Sombra Apocalíptica",
        "pasta": os.path.join(PASTA_TABELAS, "sombra_apocaliptica"),
        "tem_efeitos": True,
        "efeitos_globais": False,
        "métrica_menor_é_melhor": False,
        "coluna_alvo": "pontuacao_media",
        "colunas_cat": ["batalha", "efeito_buff", "personagem_1", "personagem_2", "personagem_3", "personagem_4"],
    },
    "pura_ficcao": {
        "nome": "Pura Ficção",
        "pasta": os.path.join(PASTA_TABELAS, "pura_ficcao"),
        "tem_efeitos": True,
        "efeitos_globais": True,
        "métrica_menor_é_melhor": False,
        "coluna_alvo": "pontuacao_media",
        "colunas_cat": ["batalha", "efeito_buff", "personagem_1", "personagem_2", "personagem_3", "personagem_4"],
    },
    "memoria_do_caos": {
        "nome": "Memória do Caos",
        "pasta": os.path.join(PASTA_TABELAS, "memoria_do_caos"),
        "tem_efeitos": False,
        "efeitos_globais": False,
        "métrica_menor_é_melhor": True,
        "coluna_alvo": "ciclos_medios",
        "colunas_cat": ["batalha", "personagem_1", "personagem_2", "personagem_3", "personagem_4"],
    },
}

# ---------------------------------------------------------------------------
# Cache em memória de CSVs e modelos treinados, invalidado por mtime do
# arquivo. Evita releitura de disco e retreino do RandomForest a cada
# requisição quando os dados não mudaram.
# ---------------------------------------------------------------------------
_CSV_CACHE = {}      # caminho -> (mtime, DataFrame)
_MODELO_CACHE = {}   # modo -> (mtime, pipeline)


def _ler_csv_obrigatorio(caminho):
    if not os.path.exists(caminho):
        return pd.DataFrame()
    mtime = os.path.getmtime(caminho)
    cache = _CSV_CACHE.get(caminho)
    if cache and cache[0] == mtime:
        return cache[1].copy()
    df = pd.read_csv(caminho)
    _CSV_CACHE[caminho] = (mtime, df)
    return df.copy()


def treinar_modelo(df, config):
    if df.empty or len(df) < 2:
        return None
    X = df[config["colunas_cat"]]
    y = df[config["coluna_alvo"]]
    preprocessador = ColumnTransformer(
        transformers=[("categorias", OneHotEncoder(handle_unknown="ignore"), config["colunas_cat"])]
    )
    pipeline = Pipeline([("preprocessador", preprocessador), ("modelo", RandomForestRegressor(n_estimators=100, random_state=42))])
    pipeline.fit(X, y)
    return pipeline


def obter_modelo(modo, config, df_res, arq_res):
    """Retorna um pipeline treinado, reaproveitando o cache enquanto o
    resultados.csv do modo não mudar de mtime."""
    mtime = os.path.getmtime(arq_res) if os.path.exists(arq_res) else None
    cache = _MODELO_CACHE.get(modo)
    if cache and mtime is not None and cache[0] == mtime:
        return cache[1]
    pipeline = treinar_modelo(df_res, config)
    if pipeline is not None and mtime is not None:
        _MODELO_CACHE[modo] = (mtime, pipeline)
    return pipeline


def _gerar_combos_validos(personagens):
    """Gera, uma única vez, todas as combinações de 4 personagens que
    incluem ao menos um curandeiro — sem passar por todas as C(n,4)
    combinações possíveis e descartar depois."""
    curandeiros_disp = [p for p in personagens if p in CURANDEIROS]
    outros_disp = [p for p in personagens if p not in CURANDEIROS]

    combos = []
    for n_cura in range(1, min(4, len(curandeiros_disp)) + 1):
        n_outros = 4 - n_cura
        if n_outros > len(outros_disp):
            continue
        for combo_cura in itertools.combinations(curandeiros_disp, n_cura):
            for combo_outros in itertools.combinations(outros_disp, n_outros):
                combos.append(tuple(sorted(combo_cura + combo_outros)))
    return combos

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dados-iniciais/<modo>")
def dados_iniciais(modo):
    config = MODOS_CONFIG.get(modo)
    if not config:
        return jsonify({"error": "Modo inválido"}), 400

    df_p = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    personagens = sorted(df_p["nome"].tolist()) if not df_p.empty else []

    df_efeitos = _ler_csv_obrigatorio(os.path.join(config["pasta"], "efeitos.csv")) if config["tem_efeitos"] else pd.DataFrame()
    efeitos = df_efeitos["nome_efeito"].tolist() if not df_efeitos.empty else []

    df_res = _ler_csv_obrigatorio(os.path.join(config["pasta"], "resultados.csv"))

    # Prepara dados do gráfico
    grafico_data = []
    if not df_res.empty:
        colunas_t = [c for c in ["tentativa_1", "tentativa_2", "tentativa_3"] if c in df_res.columns]
        media = df_res[colunas_t].mean(axis=1) if colunas_t else pd.Series(dtype=float)

        # Desvio padrão por tentativa: usa a coluna já calculada no cadastro
        # quando disponível, senão calcula na hora.
        if "desvio_padrao" in df_res.columns:
            desvio = df_res["desvio_padrao"]
        elif colunas_t:
            desvio = df_res[colunas_t].std(axis=1, ddof=0)
        else:
            desvio = pd.Series([0] * len(df_res))

        # Recorde acumulado: melhor resultado já alcançado até cada tentativa,
        # respeitando se a métrica do modo é "menor é melhor".
        recorde = media.cummin() if config["métrica_menor_é_melhor"] else media.cummax()

        # Composição da equipe formatada, usada como texto no hover do gráfico.
        colunas_p = [c for c in ["personagem_1", "personagem_2", "personagem_3", "personagem_4"] if c in df_res.columns]

        def _formatar_equipe(row):
            nomes = ", ".join(str(row[c]) for c in colunas_p if pd.notna(row.get(c)))
            if config["tem_efeitos"] and pd.notna(row.get("efeito_buff")):
                nomes += f" ({row['efeito_buff']})"
            return nomes

        equipe = df_res.apply(_formatar_equipe, axis=1) if colunas_p else pd.Series([""] * len(df_res))

        grafico_data = {
            "indices": list(df_res.index),
            "t1": df_res["tentativa_1"].tolist() if "tentativa_1" in df_res.columns else [],
            "t2": df_res["tentativa_2"].tolist() if "tentativa_2" in df_res.columns else [],
            "t3": df_res["tentativa_3"].tolist() if "tentativa_3" in df_res.columns else [],
            "media": media.tolist(),
            "desvio": desvio.tolist(),
            "recorde": recorde.tolist(),
            "batalha": df_res["batalha"].tolist() if "batalha" in df_res.columns else [],
            "equipe": equipe.tolist(),
        }

    return jsonify({
        "config": config,
        "personagens": personagens,
        "efeitos": efeitos,
        "grafico": grafico_data
    })

@app.route("/api/simular", methods=["POST"])
def simular():
    dados = request.json
    modo = dados.get("modo")
    batalha = dados.get("batalha")
    top_n = int(dados.get("top_n", 10))
    
    config = MODOS_CONFIG[modo]
    df_p = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    df_efeitos = _ler_csv_obrigatorio(os.path.join(config["pasta"], "efeitos.csv")) if config["tem_efeitos"] else pd.DataFrame()
    arq_res = os.path.join(config["pasta"], "resultados.csv")
    df_res = _ler_csv_obrigatorio(arq_res)

    pipeline = obter_modelo(modo, config, df_res, arq_res)
    if pipeline is None:
        return jsonify({"error": "Dados insuficientes para treinar a IA. Cadastre dados reais primeiro."}), 400

    personagens = sorted(df_p["nome"].tolist())
    ja_testados = set()
    if not df_res.empty and "batalha" in df_res.columns:
        df_b = df_res[df_res["batalha"] == batalha]
        ja_testados = set(tuple(sorted(row)) for row in df_b[["personagem_1", "personagem_2", "personagem_3", "personagem_4"]].values)

    # Gera as combinações válidas (com ao menos 1 curandeiro) uma única vez;
    # antes isso era refeito do zero para cada buff.
    combos_validos = [c for c in _gerar_combos_validos(personagens) if c not in ja_testados]

    buffs = df_efeitos["nome_efeito"].tolist() if config["tem_efeitos"] and not df_efeitos.empty else [None]
    linhas = [
        {
            "batalha": batalha,
            "personagem_1": combo[0], "personagem_2": combo[1], "personagem_3": combo[2], "personagem_4": combo[3],
            **({"efeito_buff": buff} if config["tem_efeitos"] else {}),
        }
        for buff in buffs
        for combo in combos_validos
    ]

    if not linhas:
        return jsonify({"error": "Nenhuma combinação inédita disponível."}), 400

    df_sim = pd.DataFrame(linhas)
    df_sim["previsto"] = pipeline.predict(df_sim[config["colunas_cat"]])
    df_sim = df_sim.sort_values("previsto", ascending=config["métrica_menor_é_melhor"]).head(top_n)

    return jsonify(df_sim.to_dict(orient="records"))

@app.route("/api/cadastrar", methods=["POST"])
def cadastrar():
    dados = request.json
    modo = dados.get("modo")
    config = MODOS_CONFIG[modo]
    
    t1, t2, t3 = float(dados["t1"]), float(dados["t2"]), float(dados["t3"])
    tentativas = [t1, t2, t3]
    media = round(sum(tentativas) / 3, 2)
    melhor_val = min(tentativas) if config["métrica_menor_é_melhor"] else max(tentativas)
    desvio = round((sum((t - media) ** 2 for t in tentativas) / 3) ** 0.5, 2)

    nova_linha = {
        "id_registro": f"e_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "batalha": dados["batalha"],
        "personagem_1": dados["p1"],
        "personagem_2": dados["p2"],
        "personagem_3": dados["p3"],
        "personagem_4": dados["p4"],
        "tentativa_1": t1,
        "tentativa_2": t2,
        "tentativa_3": t3,
        config["coluna_alvo"]: media,
        "métrica_destaque": melhor_val,
        "desvio_padrao": desvio,
    }
    if config["tem_efeitos"]:
        nova_linha["efeito_buff"] = dados.get("buff")

    arq_out = os.path.join(config["pasta"], "resultados.csv")
    df_res = _ler_csv_obrigatorio(arq_out)
    df_atualizado = pd.concat([df_res, pd.DataFrame([nova_linha])], ignore_index=True)
    os.makedirs(config["pasta"], exist_ok=True)
    df_atualizado.to_csv(arq_out, index=False)

    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)