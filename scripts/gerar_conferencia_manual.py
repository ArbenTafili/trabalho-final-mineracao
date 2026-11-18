"""Gera as tabelas da conferência manual a partir dos dados reais.

Uso: python scripts/gerar_conferencia_manual.py [--dados dados/ml-25m]

- conferencia_manual/filmes_sem_genero_mais_avaliados.csv: filmes sem gênero com pelo menos 50 avaliações,
  com o link do IMDb, para anotar o gênero conferido (decisão D2).
- conferencia_manual/titulos_repetidos_candidatos.csv: pares com mesmo título, mesmo ano e mesmo código TMDb,
  com os links das duas cópias, para decidir se são o mesmo filme (decisão D3).
As colunas em branco são preenchidas à mão. O script não sobrescreve uma tabela que já tenha anotações.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "conferencia_manual"
MIN_AVALIACOES = 50


def imdb(i):
    return f"https://www.imdb.com/title/tt{int(i):07d}/"


def tmdb(i):
    return f"https://www.themoviedb.org/movie/{int(i)}" if pd.notna(i) else ""


def salvar(df, nome, colunas_manuais):
    caminho = SAIDA / nome
    if caminho.exists():
        antigo = pd.read_csv(caminho, dtype=str).fillna("")
        if (antigo[colunas_manuais] != "").any().any():
            print(f"{caminho.name}: já tem anotações, não foi sobrescrito")
            return
    df.to_csv(caminho, index=False)
    print(f"{caminho.relative_to(RAIZ)}: {len(df)} linhas")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", default=str(RAIZ / "dados" / "ml-25m"))
    d = Path(p.parse_args().dados)
    if not (d / "ratings.csv").is_file():
        sys.exit(f"Dados não encontrados em {d}; rode scripts/baixar_dados.py")
    movies = pd.read_csv(d / "movies.csv")
    links = pd.read_csv(d / "links.csv")
    ratings = pd.read_csv(d / "ratings.csv", usecols=["userId", "movieId"])
    m = movies.merge(links, on="movieId", how="left")
    m["n_avaliacoes"] = m["movieId"].map(ratings.groupby("movieId").size()).fillna(0).astype(int)
    m["title"] = m["title"].str.strip()
    SAIDA.mkdir(exist_ok=True)

    # D2: filmes sem gênero mais avaliados
    sem = m[(m["genres"] == "(no genres listed)") & (m["n_avaliacoes"] >= MIN_AVALIACOES)]
    sem = sem.sort_values("n_avaliacoes", ascending=False)
    tabela = pd.DataFrame({
        "movieId": sem["movieId"], "titulo": sem["title"], "avaliacoes": sem["n_avaliacoes"],
        "link_imdb": sem["imdbId"].map(imdb),
        "generos_no_imdb": "", "generos_movielens": "", "observacao": "",
    })
    salvar(tabela, "filmes_sem_genero_mais_avaliados.csv", ["generos_no_imdb", "generos_movielens", "observacao"])

    # D3: títulos repetidos com o mesmo código TMDb
    rep = m[m.duplicated("title", keep=False) & m["tmdbId"].notna()]
    rep = rep.groupby("title").filter(lambda g: len(g) >= 2 and g["tmdbId"].nunique() == 1)
    usuarios = ratings[ratings["movieId"].isin(rep["movieId"])].groupby("movieId")["userId"].apply(set)
    linhas = []
    for titulo, g in rep.sort_values(["title", "n_avaliacoes"], ascending=[True, False]).groupby("title", sort=False):
        a, b = g.iloc[0], g.iloc[1]
        em_comum = len(usuarios.get(a["movieId"], set()) & usuarios.get(b["movieId"], set()))
        linhas.append({
            "titulo": titulo, "tmdb": tmdb(a["tmdbId"]),
            "movieId_mais_avaliado": a["movieId"], "avaliacoes_1": a["n_avaliacoes"], "generos_1": a["genres"],
            "imdb_1": imdb(a["imdbId"]),
            "movieId_menos_avaliado": b["movieId"], "avaliacoes_2": b["n_avaliacoes"], "generos_2": b["genres"],
            "imdb_2": imdb(b["imdbId"]),
            "usuarios_nas_duas": em_comum,
            "mesmo_filme": "", "evidencia": "", "generos_decididos": "", "observacao": "",
        })
    salvar(pd.DataFrame(linhas), "titulos_repetidos_candidatos.csv",
           ["mesmo_filme", "evidencia", "generos_decididos", "observacao"])


if __name__ == "__main__":
    main()
