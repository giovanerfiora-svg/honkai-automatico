import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Tabelas/sombra_apocaliptica/resultados.csv")

plt.figure(figsize=(14,6))

x = range(len(df))

# Tentativa 1
plt.plot(x, df["tentativa_1"], linewidth=1, alpha=0.7)
plt.scatter(x, df["tentativa_1"], s=30, label="Tentativa 1")

# Tentativa 2
plt.plot(x, df["tentativa_2"], linewidth=1, alpha=0.7)
plt.scatter(x, df["tentativa_2"], s=30, label="Tentativa 2")

# Tentativa 3
plt.plot(x, df["tentativa_3"], linewidth=1, alpha=0.7)
plt.scatter(x, df["tentativa_3"], s=30, label="Tentativa 3")

plt.xlabel("Composição")
plt.ylabel("Pontuação")
plt.title("Pontuação das Tentativas")
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()