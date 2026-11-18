"""O arquivo oficial do IMDb (title.basics) como fonte de gênero e de ano (decisões D2 e D4).

Uso: python analises/imdb.py

1. Cobertura: quantos filmes do MovieLens o IMDb encontra pelo código da tabela links.
2. Concordância de gêneros, nos filmes que têm gênero nas duas fontes, com a tabela de conversão IMDb → MovieLens.
3. Concordância de ano: startYear do IMDb × ano do título do MovieLens.
4. Teste externo do kNN: nos filmes SEM gênero no MovieLens, o gênero previsto pelo público é comparado com o
   gênero do IMDb (os filmes sem gênero não entram no treino).
"""
import numpy as np
import pandas as pd

from comum import FAIXAS, IMDB, SEM_GENERO, atributos_knn, carregar_movielens, na_faixa, prever_knn, publico_por_genero, rotulo_faixa

# Conversão IMDb → MovieLens. Os gêneros que não existem no MovieLens são ignorados.
IGUAIS = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
CONVERSAO = {**{g: g for g in IGUAIS}, "Family": "Children"}

movies, R, Y, generos = carregar_movielens()
S, n = publico_por_genero(R, Y)
movies["tconst"] = movies["imdbId"].map(lambda i: f"tt{int(i):07d}")
movies["n_avaliacoes"] = n
movies["ano_titulo"] = movies["title"].str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)

imdb = pd.read_csv(IMDB, sep="\t", usecols=["tconst", "titleType", "startYear", "genres"], dtype=str,
                   na_values="\\N", quoting=3)
imdb = imdb[imdb["tconst"].isin(set(movies["tconst"]))]
m = movies.merge(imdb, on="tconst", how="left", suffixes=("", "_imdb"))
assert len(m) == len(movies)


def converter(texto):
    if pd.isna(texto):
        return frozenset()
    return frozenset(CONVERSAO[g] for g in texto.split(",") if g in CONVERSAO)


m["generos_imdb"] = m["genres_imdb"].map(converter)
m["generos_ml"] = m["genres"].map(lambda s: frozenset() if s == SEM_GENERO else frozenset(s.split("|")) - {"IMAX"})
sem_genero = m["genres"] == SEM_GENERO
sem_ano = m["ano_titulo"].isna()

print("=== 1. Cobertura ===")
print(f"Filmes do MovieLens encontrados no IMDb: {m['titleType'].notna().sum():,} de {len(m):,}")
for nome, sel in [("sem gênero", sem_genero), ("sem gênero e com avaliação", sem_genero & (n > 0)), ("sem ano", sem_ano)]:
    print(f"{nome}: {sel.sum():,} filmes; no IMDb: {m.loc[sel, 'titleType'].notna().sum():,}; "
          f"com gênero do IMDb convertível: {(m.loc[sel, 'generos_imdb'].str.len() > 0).sum():,}; "
          f"com ano no IMDb: {m.loc[sel, 'startYear'].notna().sum():,}")
print("Tipo de título no IMDb dos filmes sem gênero:", m.loc[sem_genero, "titleType"].value_counts().head(6).to_dict())
cobertos = sem_genero & (m["generos_imdb"].str.len() > 0)
print(f"Avaliações dos filmes sem gênero que o IMDb preenche: {int(n[cobertos].sum()):,} de {int(n[sem_genero].sum()):,}")

print("\n=== 2. Concordância de gêneros (filmes com gênero nas duas fontes) ===")
ambos = (m["generos_ml"].str.len() > 0) & (m["generos_imdb"].str.len() > 0)
inter = np.array([len(a & b) for a, b in zip(m.loc[ambos, "generos_ml"], m.loc[ambos, "generos_imdb"])])
uniao = np.array([len(a | b) for a, b in zip(m.loc[ambos, "generos_ml"], m.loc[ambos, "generos_imdb"])])
print(f"Filmes comparados: {ambos.sum():,}")
print(f"Pelo menos um gênero em comum: {100 * (inter > 0).mean():.1f}%; mesmos gêneros: {100 * (inter == uniao).mean():.1f}%; "
      f"Jaccard médio: {(inter / uniao).mean():.3f}")
print("Por gênero do MovieLens: % dos filmes que o IMDb também marca com ele")
por_genero = {g: 100 * np.mean([g in b for a, b in zip(m.loc[ambos, "generos_ml"], m.loc[ambos, "generos_imdb"]) if g in a])
              for g in generos if g != "IMAX"}
print("  " + "; ".join(f"{g} {v:.0f}" for g, v in sorted(por_genero.items(), key=lambda x: -x[1])))
musica = m.loc[ambos, "genres_imdb"].str.contains("Music", na=False) & ~m.loc[ambos, "genres_imdb"].str.contains("Musical", na=False)
print(f"IMDb com Music (sem Musical): {musica.sum():,} filmes; destes, o MovieLens marca Musical em "
      f"{100 * m.loc[ambos][musica]['generos_ml'].map(lambda s: 'Musical' in s).mean():.0f}% e Documentary em "
      f"{100 * m.loc[ambos][musica]['generos_ml'].map(lambda s: 'Documentary' in s).mean():.0f}%")

print("\n=== 3. Concordância de ano (filmes com ano nas duas fontes) ===")
dois = m["ano_titulo"].notna() & m["startYear"].notna()
dif = (m.loc[dois, "startYear"].astype(float) - m.loc[dois, "ano_titulo"]).abs()
print(f"Filmes comparados: {dois.sum():,}; ano igual: {100 * (dif == 0).mean():.1f}%; diferença de até 1 ano: "
      f"{100 * (dif <= 1).mean():.1f}%; mais de 1 ano: {(dif > 1).sum():,}")

print("\n=== 4. Teste externo do kNN nos filmes sem gênero ===")
conhecido = (Y.sum(1) > 0) & (S.sum(1) > 0)
X = atributos_knn(S, conhecido)
treino = np.where(conhecido & (n >= 10))[0]
alvo = np.where(sem_genero.to_numpy() & (S.sum(1) > 0) & (m["generos_imdb"].str.len() > 0).to_numpy())[0]
pred, confianca = prever_knn(X[treino], Y[treino], X[alvo])
drama = generos.index("Drama")
acerto_knn = np.array([generos[p] in m.at[i, "generos_imdb"] for p, i in zip(pred, alvo)])
acerto_moda = np.array(["Drama" in m.at[i, "generos_imdb"] for i in alvo])
N = Y[treino].sum(0)
pred_raiz = (S[alvo] / np.sqrt(N)).argmax(1)
acerto_raiz = np.array([generos[p] in m.at[i, "generos_imdb"] for p, i in zip(pred_raiz, alvo)])
linhas = []
for a, b in FAIXAS:
    sel = na_faixa(n[alvo], a, b)
    if sel.sum():
        linhas.append({"avaliadores": rotulo_faixa(a, b), "filmes": int(sel.sum()), "chutar Drama %": 100 * acerto_moda[sel].mean(),
                       "1/√N %": 100 * acerto_raiz[sel].mean(), "kNN %": 100 * acerto_knn[sel].mean()})
print(pd.DataFrame(linhas).round(1).to_string(index=False))
