"""Gera um MovieLens sintético pequeno, com a mesma estrutura de arquivos, para o teste rápido do notebook.

Uso: python scripts/gerar_sinteticos.py <pasta_destino> [--usuarios 12000] [--filmes 3000]

Além de imitar a forma dos dados reais (popularidade desigual, meia estrela só a partir de 2003, gêneros
multirrótulo), inclui de propósito os casos de borda que o notebook precisa tratar:
- filmes sem gênero, entre eles um com exatamente 9 e outro com exatamente 10 avaliadores (limite da imputação);
- filmes sem ano no título;
- títulos repetidos: mesma cópia cadastrada duas vezes (mesmo TMDb, usuários em comum: é fundida), homônimos
  (códigos diferentes), mesmo TMDb com gêneros sem nada em comum (ficam separados) e uma cópia sem gênero;
- filmes sem gênero que o IMDb não resolve (só 'Short', sem gênero ou código ausente): só entre os pouco avaliados,
  como nos dados reais, para que a exclusão da 2.2-d passe no critério de 'não pesa';
- refilmagem (mesmo nome, anos diferentes);
- IMAX quase sempre acompanhado de outro gênero; o único filme só com IMAX fica sem gênero quando o IMAX sai (2.2)
  e recebe gênero do IMDb (2.2-b), como o filme 4460 nos dados reais.
O sinal de gênero é forte o bastante para a imputação passar da moda, para que esse trecho do código seja exercitado.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENEROS = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
           "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
PESO = np.array([.12, .09, .04, .05, .14, .07, .02, .18, .05, .01, .04, .0, .02, .03, .08, .05, .09, .02, .01])
INICIO, FIM = pd.Timestamp("1996-01-01").timestamp(), pd.Timestamp("2019-11-01").timestamp()


def gerar(destino, n_usuarios, n_filmes, semente=1):
    rng = np.random.default_rng(semente)
    peso = PESO / PESO.sum()
    imax = GENEROS.index("IMAX")

    # --- Filmes -----------------------------------------------------------------------------
    filmes, indicadora = [], np.zeros((n_filmes, len(GENEROS)))
    for i in range(n_filmes):
        ano = int(rng.integers(1920, 2020))
        titulo = f"Filme {i + 1} ({ano})" if rng.random() > 0.03 else f"Filme {i + 1}"
        if rng.random() < 0.01:
            titulo += " "
        if rng.random() < 0.08:
            generos = "(no genres listed)"
        else:
            k = min(1 + rng.poisson(0.8), 4)
            escolhidos = list(rng.choice(len(GENEROS), size=k, replace=False, p=peso))
            if ano >= 2000 and rng.random() < 0.05:  # IMAX sempre junto de outro gênero
                escolhidos.append(imax)
            indicadora[i, escolhidos] = 1
            generos = "|".join(GENEROS[j] for j in sorted(escolhidos))
        filmes.append([i + 1, titulo, generos])
    filmes[1] = [2, "Só IMAX (2015)", "IMAX"]  # o único caso de IMAX sozinho
    indicadora[1] = 0
    indicadora[1, imax] = 1

    # Títulos repetidos e refilmagem, nos últimos ids (filmes pouco populares não importam aqui).
    casos = [
        ("Cópia Duplicada (2002)", "Comedy|Crime|Drama", "Comedy|Crime|Drama|Romance"),
        ("Homônimo (2005)", "Action|Sci-Fi", "Documentary"),
        ("Mesmo TMDb Sem Interseção (2007)", "Horror|Thriller", "Drama"),
        ("Uma Cópia Sem Gênero (2010)", "Animation|Children", "(no genres listed)"),
    ]
    base = n_filmes - 2 * len(casos) - 2  # os dois últimos ids ficam para a refilmagem
    for k, (titulo, g1, g2) in enumerate(casos):
        for j, g in enumerate((g1, g2)):
            idx = base + 2 * k + j
            filmes[idx] = [idx + 1, titulo, g]
            indicadora[idx] = [1 if x in g.split("|") else 0 for x in GENEROS]
    filmes[n_filmes - 1] = [n_filmes, "Refilmagem (1960)", "Drama"]
    filmes[n_filmes - 2] = [n_filmes - 1, "Refilmagem (2015)", "Drama|Thriller"]
    movies = pd.DataFrame(filmes, columns=["movieId", "title", "genres"])

    # --- Avaliações ---------------------------------------------------------------------------
    qualidade = rng.normal(0, 0.5, n_filmes)
    popularidade = 1 / np.arange(1, n_filmes + 1) ** 1.0
    rng.shuffle(popularidade)
    popularidade[rng.random(n_filmes) < 0.05] = 0
    popularidade[base:] = np.maximum(popularidade[base:], np.median(popularidade))  # casos de borda com público
    gostos = rng.dirichlet(np.ones(len(GENEROS)) * 0.4, size=6)
    prob = []
    for t in range(len(gostos)):
        w = popularidade * (1 + 15 * indicadora @ gostos[t])  # sinal de gênero forte
        prob.append(w / w.sum())
    partes = []
    for u in range(1, n_usuarios + 1):
        n = int(min(20 + rng.lognormal(2.6, 0.9), 800))
        tipo = rng.integers(len(gostos))
        fs = rng.choice(n_filmes, size=n, replace=False, p=prob[tipo])
        vies = rng.normal(0, 0.4)
        t0 = rng.uniform(INICIO, FIM)
        duracao = 0 if rng.random() < 0.6 else rng.uniform(0, 3 * 365 * 86400)
        ts = (t0 + rng.uniform(0, max(duracao, 3600), n)).astype(np.int64)
        nota = 3.5 + vies + qualidade[fs] + rng.normal(0, 0.9, n)
        ano = pd.to_datetime(ts, unit="s").year.to_numpy()
        nota = np.where(ano < 2003, np.clip(np.round(nota), 1, 5), np.clip(np.round(nota * 2) / 2, 0.5, 5))
        partes.append(pd.DataFrame({"userId": u, "movieId": fs + 1, "rating": nota, "timestamp": ts}))
    ratings = pd.concat(partes, ignore_index=True)

    # Filmes sem gênero com exatamente 9 e 10 avaliadores (limite da imputação).
    por_usuario = ratings.groupby("userId").size()
    sem_genero = movies.loc[movies.genres == "(no genres listed)", "movieId"].to_numpy()
    sem_genero = sem_genero[sem_genero < base]
    for alvo, filme in zip((9, 10), sem_genero[:2]):
        atuais = ratings.index[ratings.movieId == filme]
        if len(atuais) > alvo:  # remove de usuários que continuam com 20 ou mais avaliações
            usuarios_ok = por_usuario[por_usuario > 25].index
            remover = [i for i in atuais if ratings.at[i, "userId"] in usuarios_ok][: len(atuais) - alvo]
            ratings = ratings.drop(remover)
        faltam = alvo - (ratings.movieId == filme).sum()
        if faltam > 0:
            ja = set(ratings.loc[ratings.movieId == filme, "userId"])
            novos = [u for u in rng.permutation(n_usuarios) + 1 if u not in ja][:faltam]
            ratings = pd.concat([ratings, pd.DataFrame({"userId": novos, "movieId": filme, "rating": 4.0,
                                                        "timestamp": int(pd.Timestamp("2010-01-01").timestamp())})])
    # Usuários que avaliaram as duas cópias do filme duplicado (mesma cópia cadastrada duas vezes).
    copia_a, copia_b = base + 1, base + 2
    ja_b = set(ratings.loc[ratings.movieId == copia_b, "userId"])
    dupla = [u for u in ratings.loc[ratings.movieId == copia_a, "userId"].unique() if u not in ja_b][:15]
    extra = pd.DataFrame({"userId": dupla, "movieId": copia_b, "rating": 4.0,
                          "timestamp": int(pd.Timestamp("2015-06-01").timestamp())})
    ratings = pd.concat([ratings, extra], ignore_index=True).sort_values(["userId", "timestamp"], kind="stable")

    # --- Tags e links -------------------------------------------------------------------------
    tags_base = ["funny", "Funny", " dark comedy", "atmospheric", "Atmospheric ", "sci-fi", "twist ending", "classic"]
    nt = 8000
    tags = pd.DataFrame({"userId": rng.integers(1, n_usuarios + 1, nt), "movieId": rng.integers(1, n_filmes + 1, nt),
                         "tag": rng.choice(tags_base, nt),
                         "timestamp": rng.integers(int(INICIO), int(FIM), nt).astype("float")})
    tags = tags.drop_duplicates(["userId", "movieId", "tag"])
    tags.loc[tags.index[:2], "timestamp"] = np.nan
    imdb = rng.choice(np.arange(10**5, 10**7), size=n_filmes, replace=False)
    tmdb = rng.choice(np.arange(1, 10**6), size=n_filmes, replace=False).astype("float")
    for k in (0, 2, 3):  # mesmo TMDb: cópia duplicada, sem interseção e uma cópia sem gênero
        tmdb[base + 2 * k + 1] = tmdb[base + 2 * k]
    tmdb[:5] = np.nan
    links = pd.DataFrame({"movieId": movies.movieId, "imdbId": imdb, "tmdbId": tmdb})

    imdb = recorte_imdb(movies, links, ratings.groupby("movieId").size(), rng)

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    movies.to_csv(destino / "movies.csv", index=False)
    ratings.to_csv(destino / "ratings.csv", index=False, float_format="%.1f")
    tags.to_csv(destino / "tags.csv", index=False, float_format="%.0f")
    links.to_csv(destino / "links.csv", index=False, float_format="%.0f")
    imdb.to_csv(destino / "imdb_recorte_movielens.tsv.gz", sep="\t", index=False, na_rep="\\N",
                compression={"method": "gzip", "mtime": 0})
    return {"filmes": len(movies), "avaliacoes": len(ratings), "usuarios": ratings.userId.nunique(), "tags": len(tags),
            "imdb": len(imdb)}


def recorte_imdb(movies, links, avaliacoes, rng, max_sem_tratamento=3):
    """Recorte do IMDb com os casos de borda: não encontrados, só 'Short', TV, anos divergentes, Family no lugar de Children.

    Os filmes sem gênero que o IMDb não resolve ficam entre os que têm até `max_sem_tratamento` avaliações."""
    linhas = []
    for (_, filme), imdb_id in zip(movies.iterrows(), links["imdbId"]):
        sem_genero = filme["genres"] == "(no genres listed)"
        pouco_avaliado = avaliacoes.get(filme["movieId"], 0) <= max_sem_tratamento
        if rng.random() < 0.005 and (not sem_genero or pouco_avaliado):  # código que o IMDb não tem mais
            continue
        titulo = filme["title"].strip()
        ano = titulo[-5:-1] if titulo.endswith(")") and titulo[-5:-1].isdigit() else str(int(rng.integers(2015, 2020)))
        if rng.random() < 0.02:
            ano = str(int(ano) + (1 if rng.random() < 0.8 else 6))
        if sem_genero:
            sorteio = rng.random() if pouco_avaliado else 1.0
            if sorteio < 0.10:
                generos = "Short"
            elif sorteio < 0.15:
                generos = None
            else:
                generos = ",".join(sorted(rng.choice([("Family" if g == "Children" else g) for g in GENEROS if g != "IMAX"], size=int(rng.integers(1, 3)),
                                                     replace=False)))
        else:
            gs = [("Family" if g == "Children" and rng.random() < 0.6 else g) for g in filme["genres"].split("|") if g != "IMAX"]
            if not gs:  # filme só com IMAX: o IMDb tem os gêneros dele (o filme 4460 real tem Adventure, Comedy e Drama)
                gs = ["Adventure", "Comedy", "Drama"]
            if rng.random() < 0.1:
                gs.append("Biography")
            generos = ",".join(sorted(gs)[:3]) or None
        tipo = "short" if generos == "Short" else ("tvEpisode" if rng.random() < 0.02 else "movie")
        linhas.append({"tconst": f"tt{int(imdb_id):07d}", "titleType": tipo, "primaryTitle": titulo.split(" (")[0],
                       "startYear": ano, "genres": generos})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("destino")
    p.add_argument("--usuarios", type=int, default=12000)
    p.add_argument("--filmes", type=int, default=3000)
    a = p.parse_args()
    print(gerar(a.destino, a.usuarios, a.filmes))
