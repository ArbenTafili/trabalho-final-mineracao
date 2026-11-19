"""Casos que continuam sem tratamento depois do IMDb e o peso de cada um (proposta 3: excluir e justificar).

Uso: python analises/casos_sem_tratamento.py

Regra proposta pelo aluno: um caso que nenhuma regra automática consegue tratar, e cujo peso é desprezível, é
excluído dos dados-alvo, com a justificativa registrada. Este script mede, para cada grupo:
A. os filmes que continuam sem gênero depois do IMDb (2.2-b): por que o IMDb não resolve, quantos o kNN validado
   (2.2-c) conseguiria tratar, quantas avaliações e quantos usuários seriam afetados;
B. os filmes que continuam sem ano;
C. os 8 pares de títulos repetidos com o mesmo código TMDb (D3): o que o IMDb diz de cada cópia.
"""
import gzip

import numpy as np
import pandas as pd

from comum import IMDB, ML, RAIZ, SEM_GENERO

RECORTE = RAIZ / "dados" / "imdb" / "imdb_recorte_movielens.tsv.gz"
IGUAIS = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
CONVERSAO = {**{g: g for g in IGUAIS}, "Family": "Children"}

movies = pd.read_csv(ML / "movies.csv").merge(pd.read_csv(ML / "links.csv"), on="movieId")
r = pd.read_csv(ML / "ratings.csv", usecols=["userId", "movieId"])
n = r["movieId"].value_counts()
movies["n_avaliacoes"] = movies["movieId"].map(n).fillna(0).astype(int)
movies["tconst"] = movies["imdbId"].map(lambda i: f"tt{int(i):07d}")
movies["ano_ml"] = movies["title"].str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)
recorte = pd.read_csv(RECORTE, sep="\t", dtype=str, na_values="\\N", keep_default_na=False, quoting=3)
m = movies.merge(recorte, on="tconst", how="left", suffixes=("", "_imdb"))
assert len(m) == len(movies)
m["generos_imdb"] = m["genres_imdb"].map(lambda s: [] if pd.isna(s) else [CONVERSAO[g] for g in s.split(",") if g in CONVERSAO])
total_aval, usuarios = len(r), r["userId"].nunique()


def efeito_usuarios(ids):
    sel = r["movieId"].isin(set(ids))
    por_usuario = sel.groupby(r["userId"]).agg(["sum", "size"])
    afetados = por_usuario[por_usuario["sum"] > 0]
    fracao = afetados["sum"] / afetados["size"]
    return (f"usuários afetados: {len(afetados):,} de {usuarios:,}; maior fração das avaliações de um usuário: "
            f"{100 * fracao.max():.1f}%; usuários que perderiam todas as avaliações: {(afetados['sum'] == afetados['size']).sum()}; "
            f"menor número de avaliações que sobra a um usuário afetado: {(afetados['size'] - afetados['sum']).min()}")


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("=== A. Filmes que continuam sem gênero depois do IMDb ===")
sem = m[(m["genres"] == SEM_GENERO) & (m["generos_imdb"].str.len() == 0)].copy()
sem["motivo"] = np.select(
    [sem["titleType"].isna(), sem["genres_imdb"].isna()],
    ["código ausente do IMDb", "IMDb sem gênero"],
    "só gêneros que não existem no MovieLens (" + sem["genres_imdb"].fillna("") + ")")
sem.loc[sem["motivo"].str.startswith("só gêneros"), "motivo"] = "só gêneros que não existem no MovieLens"
print(f"{len(sem):,} filmes, {sem['n_avaliacoes'].sum():,} avaliações ({100 * sem['n_avaliacoes'].sum() / total_aval:.4f}% do total)")
print(sem.groupby("motivo").agg(filmes=("movieId", "size"), avaliacoes=("n_avaliacoes", "sum")).to_string())
print("Gêneros do IMDb dos que só têm gêneros fora do MovieLens:",
      m.loc[sem.index][sem["motivo"] == "só gêneros que não existem no MovieLens"]["genres_imdb"].value_counts().head(8).to_dict())
faixas = pd.cut(sem["n_avaliacoes"], [-1, 0, 2, 4, 9, np.inf], labels=["0", "1–2", "3–4", "5–9", "10+"])
print("\nPor número de avaliações (o kNN da 2.2-c passa nos dois testes a partir de 5 avaliadores; na validação interna, a partir de 3):")
print(sem.groupby(faixas, observed=False).agg(filmes=("movieId", "size"), avaliacoes=("n_avaliacoes", "sum")).to_string())
com_aval = sem[sem["n_avaliacoes"] > 0]
print(f"\nSe os {len(com_aval):,} com avaliação fossem excluídos: {efeito_usuarios(com_aval['movieId'])}")
knn = sem["n_avaliacoes"] >= 5
print(f"Tratáveis pelo kNN (5 ou mais avaliadores): {knn.sum()} filmes, {sem.loc[knn, 'n_avaliacoes'].sum():,} avaliações; "
      f"sem tratamento (1 a 4): {((sem['n_avaliacoes'] > 0) & ~knn).sum()} filmes, {sem.loc[~knn, 'n_avaliacoes'].sum():,} avaliações")
print(f"Se só os de 1 a 4 avaliadores fossem excluídos: {efeito_usuarios(sem.loc[(sem['n_avaliacoes'] > 0) & ~knn, 'movieId'])}")
print("Os mais avaliados:")
print(sem.sort_values("n_avaliacoes", ascending=False)[["title", "n_avaliacoes", "titleType", "genres_imdb", "motivo"]].head(12).to_string(index=False))

print("\n=== B. Filmes que continuam sem ano depois do IMDb ===")
sem_ano = m[m["ano_ml"].isna() & m["startYear"].isna()]
print(sem_ano[["movieId", "title", "genres", "n_avaliacoes", "titleType"]].to_string(index=False))
print(f"{len(sem_ano)} filmes, {sem_ano['n_avaliacoes'].sum()} avaliações")

print("\n=== C. Títulos repetidos com o mesmo código TMDb (D3) ===")
rep = movies[movies.duplicated("title", keep=False) & movies["tmdbId"].notna()]
rep = rep.groupby("title").filter(lambda g: len(g) >= 2 and g["tmdbId"].nunique() == 1)
alvo = set(rep["tconst"])
with gzip.open(IMDB, "rt") as f:
    cab = f.readline().rstrip("\n").split("\t")
    linhas = [l.rstrip("\n").split("\t") for l in f if l[:l.index("\t")] in alvo]
imdb = pd.DataFrame(linhas, columns=cab).mask(lambda d: d == "\\N")
rep = rep.merge(imdb[["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres"]], on="tconst",
                how="left", suffixes=("", "_imdb"))
usuarios_filme = r[r["movieId"].isin(rep["movieId"])].groupby("movieId")["userId"].apply(set)
print(f"{rep['title'].nunique()} pares, {len(rep)} filmes, {rep['n_avaliacoes'].sum():,} avaliações")
for titulo, g in rep.sort_values(["title", "n_avaliacoes"], ascending=[True, False]).groupby("title", sort=False):
    a, b = g.iloc[0], g.iloc[1]
    comum = len(usuarios_filme.get(a["movieId"], set()) & usuarios_filme.get(b["movieId"], set()))
    print(f"\n{titulo} (TMDb {int(a['tmdbId'])}; usuários que avaliaram as duas cópias: {comum})")
    print(g[["movieId", "genres", "n_avaliacoes", "tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres_imdb"]].to_string(index=False))
menos = rep.sort_values("n_avaliacoes").groupby("title").head(1)
print(f"\nCópias menos avaliadas dos 8 pares: {len(menos)} filmes, {menos['n_avaliacoes'].sum():,} avaliações; {efeito_usuarios(menos['movieId'])}")
print(f"As duas cópias dos 8 pares: {len(rep)} filmes, {rep['n_avaliacoes'].sum():,} avaliações; {efeito_usuarios(rep['movieId'])}")

print("\n=== D. O kNN da 2.2-c serve para os filmes sem gênero com 5 ou mais avaliadores? ===")
# Os filmes de A com 5 ou mais avaliadores são quase todos curtas. A validação da 2.2-c foi feita em filmes com gênero
# (quase todos longas) contra a linha de base "chutar Drama". Aqui o kNN é testado em populações parecidas com a de A,
# com gênero no MovieLens: os curtas do IMDb e os títulos do IMDb que só têm gêneros que não existem no MovieLens.
# O modelo é treinado nos filmes com gênero e 10 ou mais avaliadores fora da população testada. O kNN de comum.py
# usa a proporção de gêneros do público, e acerta cerca de 3 pontos menos que o do notebook.
from comum import atributos_knn, carregar_movielens, prever_knn, publico_por_genero

print("Tipo no IMDb dos filmes de A com 5 ou mais avaliadores:", sem.loc[knn, "titleType"].fillna("código ausente").value_counts().to_dict())
filmes_k, R, Y, generos_k = carregar_movielens()
S, n_k = publico_por_genero(R, Y)
filmes_k = filmes_k.assign(tconst=filmes_k["imdbId"].map(lambda i: f"tt{int(i):07d}")).merge(
    recorte, on="tconst", how="left", suffixes=("", "_imdb"))
assert len(filmes_k) == len(Y)
conhecido_k = (Y.sum(1) > 0) & (S.sum(1) > 0)
X_k = atributos_knn(S, conhecido_k)
drama = generos_k.index("Drama")
so_fora = filmes_k["genres_imdb"].map(lambda s: pd.notna(s) and not any(g in CONVERSAO for g in s.split(","))).to_numpy()
for nome, pop in [("curtas do IMDb", conhecido_k & (filmes_k["titleType"] == "short").to_numpy()),
                  ("IMDb só com gêneros que não existem no MovieLens", conhecido_k & so_fora)]:
    treino = np.where(conhecido_k & (n_k >= 10) & ~pop)[0]
    for a, b in [(5, 9), (10, 49), (5, None)]:
        alvo_k = np.where(pop & (n_k >= a) & (n_k <= (b or np.inf)))[0]
        if not len(alvo_k):
            continue
        previsto, _ = prever_knn(X_k[treino], Y[treino], X_k[alvo_k])
        moda = Y[alvo_k].sum(0).argmax()
        print(f"  {nome}, {a}{'+' if b is None else f'–{b}'} avaliadores: {len(alvo_k):,} filmes; kNN {100 * Y[alvo_k, previsto].mean():.1f}%; "
              f"chutar Drama {100 * Y[alvo_k, drama].mean():.1f}%; chutar o gênero mais comum ({generos_k[moda]}) {100 * Y[alvo_k, moda].mean():.1f}%")
