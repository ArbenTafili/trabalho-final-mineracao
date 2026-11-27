"""PCA como no método da disciplina e o número de componentes d′ por vários critérios (pedido do professor, 26/11/2026).

Uso: python analises/pca_criterios.py

O notebook (v9) faz o PCA sobre o z-score (matriz de correlação). O método ensinado escala cada dimensão por
min-max, centraliza na média e extrai os autovalores da matriz de covariância. Este script, sobre o perfil base
(nota média, desvio-padrão da nota e a proporção de cada gênero, sem IMAX), mede:
A. os autovalores em escala (somam 1) e o d′ de cada critério:
   - % acumulada (80% e 90%);
   - broken-stick: o último autovalor que se desvia (fica acima) da distribuição broken-stick, e, para comparação,
     a regra usada até a v9 (quantos dos primeiros ficam acima antes do primeiro que fica abaixo);
   - Kaiser-Guttman com autovalores em escala pela média (λ / média dos λ ≥ 1);
   - dimensionalidade intrínseca global: dimensão de correlação (fractal, D2, pela contagem de pares), ρ-score
     (μ² / 2σ² dos pares de distância, mediana de 5 amostras) e, para comparação, a estimativa de máxima
     verossimilhança (MLE); d′ = ⌈D⌉ (o "+1" foi uma leitura errada do pedido: ver D7 em docs/decisoes.md);
B. a distribuição dos pares de distância nos dados min-max e no PCA com cada d′ (contraste e variância relativa),
   com a figura em `analises/pca_criterios_distancias.png`.
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from comum import ML, RAIZ, SEM_GENERO

AMOSTRA_DIST, AMOSTRA_ID, SEMENTE = 2000, 5000, 42
filmes = pd.read_csv(ML / "movies.csv")
filmes = filmes[filmes["genres"] != SEM_GENERO]
fg = filmes.assign(genre=filmes["genres"].str.split("|")).explode("genre")[["movieId", "genre"]]
fg = fg[fg["genre"] != "IMAX"]
r = pd.read_csv(ML / "ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
r = r[r["movieId"].isin(set(filmes["movieId"]))]
perfil = r.groupby("userId")["rating"].agg(nota_media="mean", nota_dp="std", n="size").fillna(0)
cont = r.merge(fg, on="movieId").groupby(["userId", "genre"]).size().unstack(fill_value=0)
props = cont.div(perfil["n"], axis=0).add_prefix("prop_")
X = pd.concat([perfil[["nota_media", "nota_dp"]], props], axis=1).fillna(0)
if "--com-dp-genero" in sys.argv:
    # Prévia: desvio-padrão da nota em cada gênero, encolhido para o desvio geral do usuário quando há poucas
    # avaliações no gênero: (n · dp_gênero + m · dp_geral) / (n + m), m = 5; com 0 ou 1 avaliação, vale o dp geral.
    M_ENC = 5
    rg = r.merge(fg, on="movieId")
    est = rg.groupby(["userId", "genre"])["rating"].agg(n="size", dp="std").reset_index()
    est = est.join(perfil["nota_dp"], on="userId")
    est["dp"] = est["dp"].fillna(est["nota_dp"])
    est["dp_enc"] = (est["n"] * est["dp"] + M_ENC * est["nota_dp"]) / (est["n"] + M_ENC)
    dp_g = est.pivot(index="userId", columns="genre", values="dp_enc").add_prefix("dp_")
    dp_g = dp_g.apply(lambda c: c.fillna(perfil["nota_dp"]))
    X = pd.concat([X, dp_g], axis=1).fillna(0)
X = X.loc[:, X.std() > 0]
d = X.shape[1]

# Método da disciplina: min-max por dimensão, centraliza na média, covariância, autovalores.
mm = (X - X.min()) / (X.max() - X.min())
centrado = mm - mm.mean()
autoval, autovet = np.linalg.eigh(np.cov(centrado.to_numpy(), rowvar=False))
ordem = np.argsort(autoval)[::-1]
autoval, autovet = autoval[ordem], autovet[:, ordem]
escala = autoval / autoval.sum()
acumulada = np.cumsum(escala)
broken = np.array([sum(1.0 / j for j in range(i, d + 1)) / d for i in range(1, d + 1)])
acima = escala > broken
d_broken_ultimo = int(np.where(acima)[0].max() + 1) if acima.any() else 0
d_broken_inicio = int(np.argmax(~acima)) if (~acima).any() else d
d_kaiser = int((autoval / autoval.mean() >= 1).sum())
d_80, d_90 = int(np.searchsorted(acumulada, 0.80) + 1), int(np.searchsorted(acumulada, 0.90) + 1)

# Dimensionalidade intrínseca global, numa amostra fixa dos dados min-max.
rng = np.random.default_rng(SEMENTE)
amostra = mm.to_numpy()[rng.choice(len(mm), AMOSTRA_ID, replace=False)]
dist = pdist(amostra)
dist = dist[dist > 0]
raios = np.quantile(dist, np.logspace(-3, -0.7, 25))  # do quantil 0,1% ao 20%: região de escala usada no ajuste
C = np.array([(dist <= r_).mean() for r_ in raios])
d2 = float(np.polyfit(np.log(raios), np.log(C), 1)[0])
vizinhos = NearestNeighbors(n_neighbors=21).fit(amostra).kneighbors(amostra)[0][:, 1:]
mle = float(np.mean([np.mean((k - 1) / np.maximum(np.log(vizinhos[:, [k - 1]] / vizinhos[:, :k - 1]).sum(axis=1), 1e-12))
                     for k in range(10, 21)]))
rhos = []
for s_ in range(5):
    dd_ = pdist(mm.to_numpy()[np.random.default_rng(s_).choice(len(mm), 2000, replace=False)])
    dd_ = dd_[dd_ > 0]
    rhos.append(dd_.mean() ** 2 / (2 * dd_.var()))
rho = float(np.median(rhos))
d_d2, d_mle, d_rho = math.ceil(d2), math.ceil(mle), math.ceil(rho)

pd.set_option("display.width", 200)
print(f"Perfil base: {len(X):,} usuários × {d} atributos" + (" (nota média, DP da nota, proporção e DP encolhido de cada gênero, sem IMAX)" if "--com-dp-genero" in sys.argv else f" (nota média, DP da nota e {d - 2} proporções de gênero, sem IMAX)"))
print("\n=== A. Autovalores (min-max, centralizado, covariância) ===")
tab = pd.DataFrame({"componente": range(1, d + 1), "autovalor em escala (%)": 100 * escala, "acumulada (%)": 100 * acumulada,
                    "broken-stick (%)": 100 * broken, "acima do broken-stick": acima, "λ / média dos λ": autoval / autoval.mean()})
print(tab.round(3).to_string(index=False))
print(f"\nd′ por critério:")
criterios = {
    "broken-stick, último autovalor acima": d_broken_ultimo,
    "broken-stick, regra da v9 (primeiros acima)": d_broken_inicio,
    "Kaiser-Guttman (λ / média ≥ 1)": d_kaiser,
    "80% da variância": d_80,
    "90% da variância": d_90,
    f"dimensão de correlação D2 = {d2:.2f} → ⌈D2⌉": d_d2,
    f"ρ-score = {rho:.2f} → ⌈ρ⌉": d_rho,
    f"MLE = {mle:.2f} → ⌈MLE⌉": d_mle,
}
for nome, v in criterios.items():
    print(f"  {nome}: {v} ({100 * acumulada[v - 1]:.2f}% da variância)" if v else f"  {nome}: {v}")

print("\n=== B. Distribuição dos pares de distância (amostra fixa de 2.000 usuários) ===")
idx = rng.choice(len(mm), AMOSTRA_DIST, replace=False)
espacos = {f"min-max ({d} atributos)": mm.to_numpy()[idx]}
proj = centrado.to_numpy()[idx] @ autovet
for nome, v in criterios.items():
    if v and f"PCA ({v})" not in [k.split(":")[0] for k in espacos]:
        espacos[f"PCA ({v}): {nome.split(' =')[0].split(',')[0]}"] = proj[:, :v]
linhas, distancias = [], {}
for nome, M in espacos.items():
    dd = pdist(M)
    dd = dd[dd > 0]
    distancias[nome] = dd
    linhas.append((nome, M.shape[1], (dd.max() - dd.min()) / dd.min(), dd.std() / dd.mean()))
print(pd.DataFrame(linhas, columns=["espaço", "dimensões", "contraste relativo", "variância relativa"]).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4.5))
for nome, dd in distancias.items():
    ax.hist(dd / dd.mean(), bins=80, histtype="step", density=True, linewidth=1.4, label=nome)
ax.set(title="Pares de distância (normalizados pela média): min-max e PCA com o d′ de cada critério",
       xlabel="distância / média das distâncias", ylabel="densidade")
ax.legend(fontsize=8)
ax.grid(alpha=.2)
plt.tight_layout()
nome_fig = "pca_criterios_distancias_com_dp_genero.png" if "--com-dp-genero" in sys.argv else "pca_criterios_distancias.png"
plt.savefig(RAIZ / "analises" / nome_fig, dpi=110)
print(f"\nFigura: analises/{nome_fig}")
