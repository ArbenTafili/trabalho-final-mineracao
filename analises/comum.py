"""Funções comuns às análises exploratórias (fora do notebook).

As análises desta pasta servem para decidir, com números, o que entra no notebook. Elas leem os dados reais de
`dados/` e não dependem do Spark.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
ML = RAIZ / "dados" / "ml-25m"
IMDB = RAIZ / "dados" / "imdb" / "title.basics.tsv.gz"
SEM_GENERO = "(no genres listed)"
FAIXAS = [(1, 2), (3, 4), (5, 9), (10, 49), (50, None)]


def rotulo_faixa(a, b):
    return f"{a}+" if b is None else f"{a}–{b}"


def na_faixa(n, a, b):
    return (n >= a) & (n <= (b if b is not None else np.inf))


def carregar_movielens():
    """Filmes, links e a matriz filme × usuário (1 = avaliou)."""
    movies = pd.read_csv(ML / "movies.csv").merge(pd.read_csv(ML / "links.csv"), on="movieId", how="left")
    ratings = pd.read_csv(ML / "ratings.csv", usecols=["userId", "movieId"])
    generos = sorted({g for s in movies["genres"] for g in s.split("|")} - {SEM_GENERO})
    indice_filme = {f: i for i, f in enumerate(movies["movieId"])}
    indice_usuario = {u: i for i, u in enumerate(ratings["userId"].unique())}
    R = sparse.csr_matrix((np.ones(len(ratings)), (ratings["movieId"].map(indice_filme), ratings["userId"].map(indice_usuario))),
                          shape=(len(movies), len(indice_usuario)))
    Y = np.zeros((len(movies), len(generos)))
    for i, s in enumerate(movies["genres"]):
        for g in s.split("|"):
            if g in generos:
                Y[i, generos.index(g)] = 1
    return movies, R, Y, generos


def publico_por_genero(R, Y):
    """Para cada filme, quantas avaliações o seu público deu a cada gênero, sem contar o próprio filme."""
    n_avaliadores = np.asarray(R.sum(1)).ravel()
    S = R @ (R.T @ Y) - n_avaliadores[:, None] * Y
    return S, n_avaliadores


def atributos_knn(S, conhecido):
    """Proporção de cada gênero no público do filme, dividida pela média (lift) e padronizada."""
    total = S.sum(1, keepdims=True)
    P = np.divide(S, total, out=np.zeros_like(S), where=total > 0)
    media = P[conhecido].mean(0)
    lift = P / media
    return StandardScaler().fit(lift[conhecido]).transform(lift)


def prever_knn(X_treino, Y_treino, X_teste, k=15):
    """Os k filmes com o público mais parecido votam; cada vizinho vota em todos os seus gêneros."""
    _, vizinhos = NearestNeighbors(n_neighbors=k).fit(X_treino).kneighbors(X_teste)
    votos = Y_treino[vizinhos].sum(1)
    return votos.argmax(1), votos.max(1) / k
