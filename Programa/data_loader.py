import pandas as pd

def carregar_dados():
    batalhas = pd.read_csv('batalhas.csv')
    efeitos = pd.read_csv('efeitos.csv')
    personagens = pd.read_csv('personagens.csv')
    resultados = pd.read_csv('resultados.csv')

    return batalhas, efeitos, personagens, resultados