"""Atributos por gênero no perfil do usuário: cobertura e redundância (pedido do professor, 26/11/2026).

Uso: python analises/atributos_por_genero.py

O professor pediu um desvio-padrão da nota por gênero em cada usuário, e não só o desvio-padrão geral. O aluno
abriu espaço para outros atributos do mesmo tipo. Este script mede, antes de qualquer mudança no notebook:
A. cobertura: em cada gênero, quantos usuários têm 1, 2 e 5 ou mais avaliações (a média exige 1; o desvio, 2);
B. redundância: a correlação de cada atributo por gênero com o atributo geral do usuário (nota média, desvio-padrão),
   nos usuários com 5 ou mais avaliações no gênero, e a da "afinidade" (média no gênero menos a média geral);
C. encolhimento: a média e o desvio por gênero puxados para os valores gerais do usuário quando há poucas avaliações,
   (n · valor do gênero + m · valor geral) / (n + m), com m = 5, e quanto isso muda a correlação.
Os dados são os do MovieLens original, sem a limpeza da seção 2 (o efeito dela é desprezível aqui); o IMAX fica de
fora, como o professor pediu.
"""
import numpy as np
import pandas as pd

from comum import ML, SEM_GENERO

M_ENCOLHIMENTO = 5
filmes = pd.read_csv(ML / "movies.csv")
filmes = filmes[filmes["genres"] != SEM_GENERO]
fg = filmes.assign(genre=filmes["genres"].str.split("|")).explode("genre")[["movieId", "genre"]]
fg = fg[fg["genre"] != "IMAX"]
generos = sorted(fg["genre"].unique())
fg["g"] = fg["genre"].map({g: i for i, g in enumerate(generos)}).astype("int8")
r = pd.read_csv(ML / "ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
geral = r.groupby("userId")["rating"].agg(media="mean", dp="std", n="size")
rg = r.merge(fg[["movieId", "g"]], on="movieId")
rg["r2"] = rg["rating"].astype("float64") ** 2
pg = rg.groupby(["userId", "g"]).agg(n=("rating", "size"), soma=("rating", "sum"), soma2=("r2", "sum")).reset_index()
pg["media_g"] = pg["soma"] / pg["n"]
pg["dp_g"] = np.sqrt(np.maximum(pg["soma2"] - pg["n"] * pg["media_g"] ** 2, 0) / np.maximum(pg["n"] - 1, 1))
pg.loc[pg["n"] < 2, "dp_g"] = np.nan
pg = pg.join(geral, on="userId", rsuffix="_geral")
n_usuarios = len(geral)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
linhas = []
for i, g in enumerate(generos):
    x = pg[pg["g"] == i]
    f5 = x[x["n"] >= 5]
    med_enc = (x["n"] * x["media_g"] + M_ENCOLHIMENTO * x["media"]) / (x["n"] + M_ENCOLHIMENTO)
    dp_enc = (x["n"] * x["dp_g"].fillna(x["dp"]) + M_ENCOLHIMENTO * x["dp"]) / (x["n"] + M_ENCOLHIMENTO)
    afinidade = x["media_g"] - x["media"]
    linhas.append({
        "gênero": g,
        "% usuários com 1+": 100 * len(x) / n_usuarios,
        "% com 2+ (desvio)": 100 * (x["n"] >= 2).sum() / n_usuarios,
        "% com 5+": 100 * len(f5) / n_usuarios,
        "corr média_g × média (5+)": f5["media_g"].corr(f5["media"]),
        "corr dp_g × dp (5+)": f5["dp_g"].corr(f5["dp"]),
        "corr afinidade × média (5+)": (f5["media_g"] - f5["media"]).corr(f5["media"]),
        "corr média encolhida × média (todos 1+)": med_enc.corr(x["media"]),
        "corr dp encolhido × dp (todos 1+)": dp_enc.corr(x["dp"]),
    })
tabela = pd.DataFrame(linhas).sort_values("% usuários com 1+", ascending=False)
print(f"Usuários: {n_usuarios:,}; gêneros sem IMAX: {len(generos)}; encolhimento com m = {M_ENCOLHIMENTO}")
print("\n=== A e B. Cobertura e redundância dos atributos por gênero ===")
print(tabela.round(3).to_string(index=False))
print("\nResumo das correlações (mediana entre os gêneros):")
print(tabela.drop(columns=["gênero"]).median().round(3).to_string())
