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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

PASTA_TABELAS = os.path.join(BASE_DIR, "Dados")
ARQUIVO_PERSONAGENS = os.path.join(PASTA_TABELAS, "personagens.csv")

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

_CSV_CACHE = {}      
_MODELO_CACHE = {}   

def _sanitizar_nan(valores):
    """Troca NaN por None nas listas antes de mandar pro jsonify.
    NaN "cru" vira o token inválido `NaN` no corpo do JSON, e o
    JSON.parse do navegador rejeita isso — foi a causa do gráfico
    ficar preso em 'Carregando dados...' quando havia célula vazia
    no CSV (ex.: desvio_padrao ausente em registros antigos)."""
    return [None if pd.isna(v) else v for v in valores]


def _ler_csv_obrigatorio(caminho):
    if not os.path.exists(caminho):
        return pd.DataFrame()
    mtime = os.path.getmtime(caminho)
    cache = _CSV_CACHE.get(caminho)
    if cache and cache[0] == mtime:
        return cache[1].copy()
    try:
        df = pd.read_csv(caminho)
    except Exception:
        return pd.DataFrame()
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
    mtime = os.path.getmtime(arq_res) if os.path.exists(arq_res) else None
    cache = _MODELO_CACHE.get(modo)
    if cache and mtime is not None and cache[0] == mtime:
        return cache[1]
    pipeline = treinar_modelo(df_res, config)
    if pipeline is not None and mtime is not None:
        _MODELO_CACHE[modo] = (mtime, pipeline)
    return pipeline

def _gerar_combos_validos(personagens):
    """Gera todas as combinações possíveis de 4 personagens, sem exigir curandeiro.
    A IA (e o usuário) ficam livres para testar qualquer composição de equipe."""
    if len(personagens) < 4:
        return []
    return list(itertools.combinations(sorted(personagens), 4))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dados-iniciais/<modo>")
def dados_iniciais(modo):
    config = MODOS_CONFIG.get(modo)
    if not config:
        return jsonify({"error": "Modo inválido"}), 400

    config_publica = {k: v for k, v in config.items() if k != "pasta"}

    df_p = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    personagens = sorted(df_p["nome"].tolist()) if not df_p.empty and "nome" in df_p.columns else []

    df_efeitos = _ler_csv_obrigatorio(os.path.join(config["pasta"], "efeitos.csv")) if config["tem_efeitos"] else pd.DataFrame()
    efeitos = df_efeitos["nome_efeito"].tolist() if not df_efeitos.empty and "nome_efeito" in df_efeitos.columns else []

    df_res = _ler_csv_obrigatorio(os.path.join(config["pasta"], "resultados.csv"))

    grafico_data = []
    if not df_res.empty:
        colunas_t = [c for c in ["tentativa_1", "tentativa_2", "tentativa_3"] if c in df_res.columns]
        media = df_res[colunas_t].mean(axis=1) if colunas_t else pd.Series(dtype=float)

        if "desvio_padrao" in df_res.columns:
            desvio = df_res["desvio_padrao"]
        elif colunas_t:
            desvio = df_res[colunas_t].std(axis=1, ddof=0)
        else:
            desvio = pd.Series([0] * len(df_res))

        recorde = media.cummin() if config["métrica_menor_é_melhor"] else media.cummax()

        colunas_p = [c for c in ["personagem_1", "personagem_2", "personagem_3", "personagem_4"] if c in df_res.columns]

        def _formatar_equipe(row):
            nomes = ", ".join(str(row[c]) for c in colunas_p if pd.notna(row.get(c)))
            if config["tem_efeitos"] and pd.notna(row.get("efeito_buff")):
                nomes += f" ({row['efeito_buff']})"
            return nomes

        equipe = df_res.apply(_formatar_equipe, axis=1) if colunas_p else pd.Series([""] * len(df_res))

        grafico_data = {
            "indices": list(df_res.index),
            "t1": _sanitizar_nan(df_res["tentativa_1"].tolist()) if "tentativa_1" in df_res.columns else [],
            "t2": _sanitizar_nan(df_res["tentativa_2"].tolist()) if "tentativa_2" in df_res.columns else [],
            "t3": _sanitizar_nan(df_res["tentativa_3"].tolist()) if "tentativa_3" in df_res.columns else [],
            "media": _sanitizar_nan(media.tolist()),
            "desvio": _sanitizar_nan(desvio.tolist()),
            "recorde": _sanitizar_nan(recorde.tolist()),
            "batalha": df_res["batalha"].tolist() if "batalha" in df_res.columns else [],
            "equipe": equipe.tolist(),
        }

    return jsonify({
        "config": config_publica,
        "personagens": personagens,
        "efeitos": efeitos,
        "grafico": grafico_data
    })

@app.route("/api/simular", methods=["POST"])
def simular():
    dados = request.json or {}
    modo = dados.get("modo")
    if not modo or modo not in MODOS_CONFIG:
        return jsonify({"error": "Modo inválido ou não especificado."}), 400

    batalha = dados.get("batalha", "1ª Batalha")
    try:
        top_n = int(dados.get("top_n", 10))
    except (ValueError, TypeError):
        top_n = 10

    config = MODOS_CONFIG[modo]
    df_p = _ler_csv_obrigatorio(ARQUIVO_PERSONAGENS)
    if df_p.empty or "nome" not in df_p.columns:
        return jsonify({"error": "Cadastre personagens válidos no CSV antes de simular."}), 400

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
        if not df_b.empty and all(col in df_b.columns for col in ["personagem_1", "personagem_2", "personagem_3", "personagem_4"]):
            ja_testados = set(tuple(sorted(row)) for row in df_b[["personagem_1", "personagem_2", "personagem_3", "personagem_4"]].values)

    combos_validos = [c for c in _gerar_combos_validos(personagens) if c not in ja_testados]

    buffs = df_efeitos["nome_efeito"].tolist() if config["tem_efeitos"] and not df_efeitos.empty and "nome_efeito" in df_efeitos.columns else [None]
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
    dados = request.json or {}
    modo = dados.get("modo")
    if not modo or modo not in MODOS_CONFIG:
        return jsonify({"error": "Modo inválido."}), 400

    config = MODOS_CONFIG[modo]
    
    try:
        t1, t2, t3 = float(dados["t1"]), float(dados["t2"]), float(dados["t3"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Dados de tentativas inválidos."}), 400

    tentativas = [t1, t2, t3]
    media = round(sum(tentativas) / 3, 2)
    melhor_val = min(tentativas) if config["métrica_menor_é_melhor"] else max(tentativas)
    desvio = round((sum((t - media) ** 2 for t in tentativas) / 3) ** 0.5, 2)

    nova_linha = {
        "id_registro": f"e_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "batalha": dados.get("batalha", "1ª Batalha"),
        "personagem_1": dados.get("p1"),
        "personagem_2": dados.get("p2"),
        "personagem_3": dados.get("p3"),
        "personagem_4": dados.get("p4"),
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

    # Invalida os caches do arquivo alterado para forçar releitura e retreino do modelo
    _CSV_CACHE.pop(arq_out, None)
    _MODELO_CACHE.pop(modo, None)

    return jsonify({"success": True})

if __name__ == "__main__":
    # threaded=True é essencial aqui: sem isso o servidor de desenvolvimento
    # atende UMA requisição por vez, então uma simulação pesada (que treina/
    # prediz com o RandomForest) bloqueia até o carregamento do gráfico,
    # que fica parecendo travado em "Carregando dados..." indefinidamente.
    app.run(debug=True, port=5000, threaded=True)