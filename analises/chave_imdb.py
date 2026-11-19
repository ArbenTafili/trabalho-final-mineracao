"""Conferência automática da chave MovieLens → IMDb (substitui a conferência filme a filme da decisão D2).

Uso: python analises/chave_imdb.py

O gênero e o ano que faltam no MovieLens vêm do IMDb pelo código da tabela links. Se o código aponta para outro
título, o gênero que veio junto também está errado. A conferência compara três sinais, os mesmos que uma pessoa
olharia à mão, mas em todos os filmes:

1. título: igual depois de padronizar (minúsculas, sem acento, "&" = "and", sem pontuação, artigo do fim do título
   do MovieLens levado para o início, título alternativo entre parênteses), contido no outro ou com até 2 letras de
   diferença; senão, diferente;
2. ano: igual, 1 ano de diferença, mais de 1 ano ou sem ano no MovieLens;
3. tipo do título no IMDb: filme (movie, tvMovie, video), curta, série, episódio, especial ou outro.

Os sinais são validados nos filmes que têm gênero nas duas fontes: se um sinal indica chave errada, esses filmes
devem ter menos gêneros em comum entre MovieLens e IMDb.
"""
import gzip
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from comum import IMDB, ML, SEM_GENERO

# Mesma conversão IMDb → MovieLens da seção 2.2-b (imdb.py roda a análise inteira ao ser importado).
IGUAIS = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
CONVERSAO = {**{g: g for g in IGUAIS}, "Family": "Children"}

ARTIGOS = r"The|A|An|Les|Le|La|L'|Il|Lo|I|Gli|El|Los|Las|Der|Die|Das|Ein|Eine|De|Het|Een|Un|Une|O|Os|As|Det|Den|En|Et"
TIPO = {"movie": "filme", "tvMovie": "filme", "video": "filme", "short": "curta", "tvShort": "curta",
        "tvSeries": "série", "tvMiniSeries": "série", "tvEpisode": "episódio", "tvSpecial": "especial"}


def normalizar(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def variantes_movielens(titulo):
    """Formas do título do MovieLens: sem o ano, com o artigo do fim levado ao início e com o título alternativo."""
    t = re.sub(r"\s*\(\d{4}(?:[–-]\d{0,4})?\)\s*$", "", titulo.strip())
    partes = [t]
    alternativo = re.match(r"^(.*?)\s*\((?:a\.k\.a\.\s*)?([^()]+)\)\s*$", t)
    if alternativo:
        partes = [alternativo.group(1), alternativo.group(2)]
    formas = set()
    for p in partes:
        formas.add(p)
        artigo = re.match(rf"^(.*), ({ARTIGOS})$", p.strip())
        if artigo:
            formas.add(f"{artigo.group(2)} {artigo.group(1)}")
    return {normalizar(f) for f in formas if normalizar(f)}


def distancia(a, b):
    """Número mínimo de letras trocadas, inseridas ou apagadas para ir de a até b."""
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def comparar_titulo(formas_ml, titulos_imdb):
    titulos_imdb = {normalizar(t) for t in titulos_imdb if isinstance(t, str)}
    if formas_ml & titulos_imdb:
        return "igual"
    pares = [(a, b) for a in formas_ml for b in titulos_imdb]
    if any(len(a) >= 8 and len(b) >= 8 and abs(len(a) - len(b)) <= 2 and distancia(a, b) <= 2 for a, b in pares):
        return "quase igual"
    if any(f" {a} " in f" {b} " or f" {b} " in f" {a} " for a, b in pares):
        return "contido"
    return "diferente"


def faixa_ano(ml, imdb):
    if np.isnan(ml):
        return "sem ano no MovieLens"
    if np.isnan(imdb):
        return "sem ano no IMDb"
    d = abs(ml - imdb)
    return "igual" if d == 0 else ("1 ano" if d == 1 else "mais de 1 ano")


movies = pd.read_csv(ML / "movies.csv").merge(pd.read_csv(ML / "links.csv"), on="movieId")
n = pd.read_csv(ML / "ratings.csv", usecols=["movieId"])["movieId"].value_counts()
movies["n_avaliacoes"] = movies["movieId"].map(n).fillna(0).astype(int)
movies["tconst"] = movies["imdbId"].map(lambda i: f"tt{int(i):07d}")
movies["ano_ml"] = movies["title"].str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)

md5 = hashlib.md5()
with open(IMDB, "rb") as f:
    for bloco in iter(lambda: f.read(1 << 20), b""):
        md5.update(bloco)
print(f"Arquivo do IMDb lido: {IMDB.name}, MD5 {md5.hexdigest()} (o completo: o sinal de título usa originalTitle, "
      f"que o recorte do notebook não tem)\n")

alvo = set(movies["tconst"])
with gzip.open(IMDB, "rt") as f:
    cab = f.readline().rstrip("\n").split("\t")
    linhas = [l.rstrip("\n").split("\t") for l in f if l[:l.index("\t")] in alvo]
imdb = pd.DataFrame(linhas, columns=cab).replace("\\N", np.nan)
m = movies.merge(imdb[["tconst", "titleType", "primaryTitle", "originalTitle", "startYear", "genres"]],
                 on="tconst", how="left", suffixes=("", "_imdb"))
assert len(m) == len(movies)
m = m[m["titleType"].notna()].copy()

m["formas"] = m["title"].map(variantes_movielens)
m["titulo_so_principal"] = [comparar_titulo(f, [p]) for f, p in zip(m["formas"], m["primaryTitle"])]
m["titulo"] = [comparar_titulo(f, [p, o]) for f, p, o in zip(m["formas"], m["primaryTitle"], m["originalTitle"])]
m["ano"] = [faixa_ano(a, b) for a, b in zip(m["ano_ml"], m["startYear"].astype(float))]
m["tipo"] = m["titleType"].map(TIPO).fillna("outro")
m["generos_imdb"] = m["genres_imdb"].map(lambda s: frozenset() if pd.isna(s) else frozenset(CONVERSAO[g] for g in s.split(",") if g in CONVERSAO))
m["generos_ml"] = m["genres"].map(lambda s: frozenset() if s == SEM_GENERO else frozenset(s.split("|")) - {"IMAX"})
ambos = (m["generos_ml"].str.len() > 0) & (m["generos_imdb"].str.len() > 0)
m["concorda"] = [len(a & b) > 0 for a, b in zip(m["generos_ml"], m["generos_imdb"])]
preenchidos = (m["genres"] == SEM_GENERO) & (m["generos_imdb"].str.len() > 0)


def tabela(coluna):
    ctl = m[ambos].groupby(coluna).agg(filmes_controle=("concorda", "size"), concordancia_pct=("concorda", "mean"))
    ctl["concordancia_pct"] = (100 * ctl["concordancia_pct"]).round(1)
    alv = m[preenchidos].groupby(coluna).agg(filmes_preenchidos=("movieId", "size"), avaliacoes=("n_avaliacoes", "sum"))
    return ctl.join(alv, how="outer").fillna(0).astype({"filmes_controle": int, "filmes_preenchidos": int, "avaliacoes": int})


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(f"Filmes com gênero nas duas fontes (controle): {ambos.sum():,}; pelo menos um gênero em comum: {100 * m.loc[ambos, 'concorda'].mean():.1f}%")
print(f"Filmes sem gênero preenchidos pelo IMDb (alvo): {preenchidos.sum():,} ({m.loc[preenchidos, 'n_avaliacoes'].sum():,} avaliações)\n")
print("=== 1. Título (com primaryTitle e originalTitle) ===")
print(tabela("titulo"))
print("\nSó com primaryTitle (o que o recorte atual tem): ",
      m[ambos]["titulo_so_principal"].value_counts().to_dict(), "| alvo:", m[preenchidos]["titulo_so_principal"].value_counts().to_dict())
print("\n=== 2. Ano ===")
print(tabela("ano"))
print("\n=== 3. Tipo no IMDb ===")
print(tabela("tipo"))
print("\n=== 4. Título × ano ===")
print(tabela(["titulo", "ano"]))
print("\n=== 5. Curtas e episódios por número de avaliações (alvo) ===")
for t in ["curta", "episódio"]:
    sel = preenchidos & (m["tipo"] == t)
    print(t, pd.cut(m.loc[sel, "n_avaliacoes"], [0, 9, 49, 99, np.inf], labels=["1–9", "10–49", "50–99", "100+"]).value_counts().sort_index().to_dict())
print("\n=== 6. Exemplos do alvo com título diferente ou ano com mais de 1 ano de diferença ===")
ex = m[preenchidos & ((m["titulo"] == "diferente") | (m["ano"] == "mais de 1 ano"))]
print(ex.sort_values("n_avaliacoes", ascending=False)[["title", "n_avaliacoes", "primaryTitle", "originalTitle", "startYear", "titleType", "titulo", "ano"]].head(25).to_string())

print("\n=== 7. Referência: concordância se a chave fosse sorteada ===")
ctl = m[ambos]
rng = np.random.default_rng(42)
sorteadas = []
for _ in range(10):
    embaralhado = ctl["generos_imdb"].to_numpy()[rng.permutation(len(ctl))]
    sorteadas.append(100 * np.mean([len(a & b) > 0 for a, b in zip(ctl["generos_ml"], embaralhado)]))
base, ok = np.mean(sorteadas), 100 * ctl["concorda"].mean()
print(f"Chave verdadeira: {ok:.1f}%; chave sorteada (10 sorteios, semente 42): {base:.1f}% ± {np.std(sorteadas):.1f}")
print("Fração estimada de chaves erradas em cada grupo = (concordância geral − concordância do grupo) / (geral − sorteada):")
for nome, sel in [("sem ano no MovieLens", ctl["ano"] == "sem ano no MovieLens"),
                  ("título diferente e ano com mais de 1 ano", (ctl["titulo"] == "diferente") & (ctl["ano"] == "mais de 1 ano")),
                  ("título diferente", ctl["titulo"] == "diferente"), ("curta", ctl["tipo"] == "curta"), ("série", ctl["tipo"] == "série")]:
    c = 100 * ctl.loc[sel, "concorda"].mean()
    print(f"  {nome}: {sel.sum():,} filmes; concordância {c:.1f}%; chaves erradas estimadas: {max(0, (ok - c) / (ok - base)):.0%}")
print("\nFilmes sem ano no MovieLens, por tipo no IMDb (controle):")
sa = ctl[ctl["ano"] == "sem ano no MovieLens"]
print(sa.groupby("tipo")["concorda"].agg(["size", "mean"]).round(3).to_string())
print("\nFilmes sem ano no MovieLens e sem gênero em comum (controle):")
print(sa[~sa["concorda"]][["title", "genres", "primaryTitle", "startYear", "titleType", "genres_imdb", "n_avaliacoes"]].to_string())

print("\n=== 8. Margem de erro dos gêneros preenchidos pelo IMDb ===")
# Estimativa sob a hipótese de que uma chave errada concorda na taxa da chave sorteada: se uma fração f das chaves
# está errada, a concordância é (1 − f)·(acerto da chave certa) + f·(sorteada) ≥ 1 − f·(1 − sorteada), então
# f ≤ (1 − concordância) / (1 − sorteada). Uma chave errada que aponta para um título parecido (episódio, homônimo)
# concorda mais que a sorteada, então f pode ser maior; em compensação, ela estraga menos o gênero.
# O controle comparável são os filmes que entraram depois do ML-20M (movieId > 131.262), como 90% dos preenchidos.
MAIOR_ID_ML20M = 131_262


def sorteada(df, vezes=10):
    gerador = np.random.default_rng(42)
    return np.mean([np.mean([len(a & b) > 0 for a, b in zip(df["generos_ml"], df["generos_imdb"].to_numpy()[gerador.permutation(len(df))])])
                    for _ in range(vezes)])


def wilson(acertos, total, z=1.96):
    p = acertos / total
    centro, meia = (p + z * z / (2 * total)) / (1 + z * z / total), z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centro - meia, centro + meia


alvo_g = m[preenchidos]
filmes_alvo, aval_alvo, total_aval = len(alvo_g), int(alvo_g["n_avaliacoes"].sum()), int(n.sum())
recentes_alvo = (alvo_g["movieId"] > MAIOR_ID_ML20M).mean()
comp = ctl[ctl["movieId"] > MAIOR_ID_ML20M]
base_comp, ok_comp = sorteada(comp), comp["concorda"].mean()
print(f"Filmes preenchidos que entraram depois do ML-20M: {100 * recentes_alvo:.0f}%")
print(f"Controle comparável (movieId > {MAIOR_ID_ML20M:,}): {len(comp):,} filmes; concordância {100 * ok_comp:.2f}%; "
      f"chave sorteada {100 * base_comp:.2f}%; sem nenhum gênero em comum: {100 * (1 - ok_comp):.2f}%")
estr = comp.groupby("ano")["concorda"].agg(concordancia="mean", filmes_controle="size")
estr["chaves_erradas"] = ((1 - estr["concordancia"]) / (1 - base_comp)).clip(0, 1)
peso = alvo_g.groupby("ano").agg(filmes=("movieId", "size"), avaliacoes=("n_avaliacoes", "sum"))
estr = peso.join(estr, how="left")
estr["chaves_erradas"] = estr["chaves_erradas"].fillna((1 - ok_comp) / (1 - base_comp))
print(estr.round(3).to_string())
f_filmes = (estr["chaves_erradas"] * estr["filmes"]).sum()
f_aval = (estr["chaves_erradas"] * estr["avaliacoes"]).sum()
print(f"Estimativa de chaves erradas nos preenchidos: {100 * f_filmes / filmes_alvo:.1f}% dos filmes ({f_filmes:,.0f} de {filmes_alvo:,}) "
      f"e {100 * f_aval / aval_alvo:.1f}% das avaliações ({f_aval:,.0f} de {aval_alvo:,}; {100 * f_aval / total_aval:.4f}% de todas as {total_aval:,})")
print(f"Conta simples no controle inteiro, para comparação: (100 − {ok:.2f}) / (100 − {base:.2f}) = {100 * (100 - ok) / (100 - base):.2f}%")

print("\n=== 9. Margem de erro dos anos preenchidos pelo IMDb ===")
sem_ano_ml = m["ano"] == "sem ano no MovieLens"
anos = m[sem_ano_ml & m["startYear"].notna()]
c_sa = ctl[ctl["ano"] == "sem ano no MovieLens"]
lo, hi = wilson(c_sa["concorda"].sum(), len(c_sa))
f_sa, f_sa_hi = (1 - c_sa["concorda"].mean()) / (1 - base / 100), (1 - lo) / (1 - base / 100)
print(f"Anos preenchidos (filmes sem ano no MovieLens e com ano no IMDb): {len(anos):,} filmes, {anos['n_avaliacoes'].sum():,} avaliações")
print(f"Controle sem ano no MovieLens: {c_sa['concorda'].sum()} de {len(c_sa)} concordam ({100 * c_sa['concorda'].mean():.1f}%; IC 95% de Wilson {100 * lo:.1f}% a {100 * hi:.1f}%)")
print(f"Chaves erradas estimadas nesse grupo: {100 * f_sa:.1f}% (até {100 * f_sa_hi:.1f}% no limite do IC) → "
      f"cerca de {f_sa * len(anos):,.0f} anos possivelmente errados (até {f_sa_hi * len(anos):,.0f})")

print("\n=== 10. Efeito por usuário ===")
r = pd.read_csv(ML / "ratings.csv", usecols=["userId", "movieId"])
r["preenchido"] = r["movieId"].isin(set(alvo_g["movieId"]))
por_usuario = r.groupby("userId")["preenchido"].mean()
taxa_aval = f_aval / aval_alvo
print(f"Usuários: {len(por_usuario):,}; com pelo menos uma avaliação de filme preenchido: {(por_usuario > 0).sum():,}")
print(f"Maior fração das avaliações de um usuário em filmes preenchidos: {100 * por_usuario.max():.1f}%; "
      f"usuários acima de 5%: {(por_usuario > 0.05).sum():,}; acima de 10%: {(por_usuario > 0.10).sum():,}")
print(f"Efeito esperado no usuário mais exposto: {100 * por_usuario.max():.1f}% × {100 * taxa_aval:.1f}% = "
      f"{100 * por_usuario.max() * taxa_aval:.2f} ponto percentual das avaliações dele com gênero possivelmente errado")

print("\n=== 11. Os filmes sem gênero com 50 ou mais avaliações (substitui a conferência manual) ===")
mais = movies[(movies["genres"] == SEM_GENERO) & (movies["n_avaliacoes"] >= 50)]
no_imdb = m[m["movieId"].isin(mais["movieId"])].copy()
passa = (no_imdb["titulo"] == "igual") & no_imdb["ano"].isin(["igual", "1 ano"]) & (no_imdb["tipo"] == "filme")
no_imdb["gênero preenchido"] = no_imdb["movieId"].isin(set(alvo_g["movieId"]))
sep = no_imdb[~passa].sort_values("n_avaliacoes", ascending=False)
# Outros filmes do MovieLens com o mesmo título: igual depois de padronizar, ou um título é o começo do outro
# (o mais curto com pelo menos 2 palavras). Mostram, por exemplo, o longa que tem o nome de um curta, ou as outras
# partes de uma série.
formas_todas = movies.assign(formas=movies["title"].map(variantes_movielens))[["movieId", "title", "formas"]]


def comeco(curto, longo):
    return len(curto.split()) >= 2 and longo.startswith(curto + " ")


def mesmo_titulo(linha):
    achados = [t for mid, t, fs in formas_todas.itertuples(index=False) if mid != linha["movieId"] and any(
        a == b or comeco(a, b) or comeco(b, a) for a in linha["formas"] for b in fs)]
    return "; ".join(achados) if achados else "—"


sep["outros no MovieLens"] = sep.apply(mesmo_titulo, axis=1)
print(f"{len(mais)} filmes; no IMDb: {len(no_imdb)}; confirmados pela regra (título igual, ano igual ou com 1 ano de diferença, "
      f"tipo filme): {passa.sum()}; separados: {len(sep)}, dos quais {sep['gênero preenchido'].sum()} com gênero do IMDb "
      f"({sep.loc[sep['gênero preenchido'], 'n_avaliacoes'].sum():,} avaliações)")
print(sep[["title", "n_avaliacoes", "primaryTitle", "startYear", "titleType", "genres_imdb", "titulo", "ano", "gênero preenchido"]].to_string(index=False))
print("\nOutros filmes do MovieLens com o mesmo título (igual depois de padronizar, ou um é o começo do outro):")
for titulo, outros in zip(sep["title"], sep["outros no MovieLens"]):
    print(f"  {titulo}: {outros}")
